# SPDX-License-Identifier: Apache-2.0
"""A request admitted right after another one finished, while the finished
request's KV still sits in the async store-cache pipeline.

C7d (46 GB, image-prefill): the third request reached the in-stream preflight
0.1 s after the second finished. Nothing was admitted, but the store-cache
worker still owned the finished request's extracted KV (~4 GB over the
weights), so ``current + KV + transient`` crossed the line and the request
was rejected by the prefill memory guard although it fits alone. That memory
comes back when the write drains: the request must wait at the head (as it
does beside admitted rows) and run once the pipeline gives the memory back.

The tests drive the real scheduler loop with its real async store-cache
worker, held mid-write by an event; the usage reading carries the residual
only while the pipeline still owns the payload.
"""

import threading

from test_scheduler_capacity_gate import GB, _guarded_rig
from test_scheduler_park import PROMPT_A, PROMPT_C, submit, waiting_ids

import omlx.scheduler as scheduler_module


def _step(rig, n=1):
    """Step without waiting for the (blocked) store-cache worker."""
    outputs = []
    scheduler = rig.scheduler
    for _ in range(n):
        outputs.extend(scheduler.step().outputs)
        commit_executor = getattr(scheduler, "_prefill_commit_executor", None)
        if commit_executor is not None:
            commit_executor.submit(lambda: None).result(timeout=20)
        scheduler._process_finished_commits()
    return outputs


def _residual_rig(tmp_path, tokenizer, monkeypatch):
    """One request finishes and its store-cache write is held in flight.

    Returns the rig, the event that lets the write finish, and a cell whose
    first item is the usage the scheduler reads while the payload is held
    (the admission line: no room). Once the pipeline releases the payload the
    usage drops to 50 GB under the line."""
    rig, _usage = _guarded_rig(
        tmp_path, tokenizer, monkeypatch, weights_gb=20, free_gb=50
    )
    scheduler = rig.scheduler
    scheduler._prefill_memory_guard = True
    # Production wiring: the gate that counts post-completion store jobs.
    scheduler._store_cache_gate = scheduler_module._StoreCacheGate(cap=3)
    # No idle model to evict: the eviction ladder cannot help.
    monkeypatch.setattr(
        scheduler, "_raise_prefill_eviction_if_available", lambda **_kw: None
    )
    release = threading.Event()
    real_worker = scheduler._async_store_cache_worker

    def held_worker(*args, **kwargs):
        # Only the post-completion store is held; prefill progress commits
        # reuse the same worker on their own executor and run normally.
        if threading.current_thread().name.startswith("omlx-store-cache"):
            release.wait(timeout=30)
        return real_worker(*args, **kwargs)

    monkeypatch.setattr(scheduler, "_async_store_cache_worker", held_worker)

    limit = scheduler._admission_limit_bytes()
    held_usage = [limit]

    def usage(**_kw):
        gate = scheduler._store_cache_gate
        owned = (gate is not None and gate.in_flight > 0) or bool(
            scheduler._pending_async_removes
        )
        return held_usage[0] if owned else limit - 50 * GB

    monkeypatch.setattr(scheduler, "_current_usage_bytes", usage)

    first = submit(rig, "first", PROMPT_A, arrival=0.5, max_tokens=2)
    for _ in range(60):
        if first.is_finished():
            break
        _step(rig)
    assert first.is_finished() and first.finish_reason != "error"
    assert scheduler._num_admitted_requests() == 0
    assert scheduler._store_cache_gate.in_flight == 1
    return rig, release, held_usage


def test_a_request_behind_a_draining_store_write_waits_instead_of_failing(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, release, _held = _residual_rig(tmp_path, mock_tokenizer, monkeypatch)
    try:
        scheduler = rig.scheduler
        second = submit(rig, "second", PROMPT_C, arrival=1.0)
        outputs = _step(rig, 4)
        assert not [o for o in outputs if o.error], [o.error for o in outputs]
        assert waiting_ids(rig) == ["second"]
        assert "second" in scheduler.requests

        # The write finishes; the pipeline gives the payload back.
        release.set()
        for _ in range(200):
            if second.is_finished():
                break
            outputs.extend(_step(rig))
            scheduler._store_cache_executor.submit(lambda: None).result(timeout=20)
        assert second.is_finished()
        assert second.finish_reason != "error"
        assert not [o for o in outputs if o.error]
    finally:
        release.set()
        rig.close()


def test_nothing_overtakes_the_head_waiting_for_the_store_write(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, release, _held = _residual_rig(tmp_path, mock_tokenizer, monkeypatch)
    try:
        submit(rig, "second", PROMPT_C + PROMPT_A[:12], arrival=1.0)
        submit(rig, "newer", PROMPT_A, arrival=2.0)
        outputs = _step(rig, 4)
        assert not [o for o in outputs if o.error]
        assert waiting_ids(rig) == ["second", "newer"]
    finally:
        release.set()
        rig.close()


def test_the_waiting_head_is_not_retried_every_step_while_the_write_holds(
    tmp_path, mock_tokenizer, monkeypatch
):
    monkeypatch.setattr(
        scheduler_module, "DEFERRED_ADMISSION_RETRY_BASE_S", 60.0, raising=False
    )
    monkeypatch.setattr(
        scheduler_module, "DEFERRED_ADMISSION_RETRY_MAX_S", 60.0, raising=False
    )
    rig, release, _held = _residual_rig(tmp_path, mock_tokenizer, monkeypatch)
    try:
        scheduler = rig.scheduler
        attempts = []
        real_check = scheduler._preflight_memory_check

        def check(request):
            if request.request_id == "second":
                attempts.append(1)
            return real_check(request)

        monkeypatch.setattr(scheduler, "_preflight_memory_check", check)
        submit(rig, "second", PROMPT_C, arrival=1.0)
        _step(rig, 8)
        assert len(attempts) == 1, attempts
    finally:
        release.set()
        rig.close()


def test_without_a_pending_store_write_the_old_rejection_still_applies(
    tmp_path, mock_tokenizer, monkeypatch
):
    """Memory nothing will give back (another app, a leak) is not waited on:
    with nothing admitted and nothing draining, the guard rejects as before."""
    rig, usage = _guarded_rig(
        tmp_path, mock_tokenizer, monkeypatch, weights_gb=20, free_gb=50
    )
    try:
        scheduler = rig.scheduler
        scheduler._prefill_memory_guard = True
        scheduler._store_cache_gate = scheduler_module._StoreCacheGate(cap=3)
        monkeypatch.setattr(
            scheduler, "_raise_prefill_eviction_if_available", lambda **_kw: None
        )
        usage[0] = scheduler._admission_limit_bytes()
        assert scheduler._store_cache_gate.in_flight == 0
        submit(rig, "only", PROMPT_C, arrival=1.0)
        outputs = _step(rig, 2)
        errors = [o for o in outputs if o.request_id == "only" and o.error]
        assert len(errors) == 1
        assert errors[0].finish_reason == "error"
        assert "only" not in scheduler.requests
    finally:
        rig.close()


def test_a_pending_store_write_makes_a_fits_alone_request_fit_later(
    tmp_path, mock_tokenizer, monkeypatch
):
    rig, release, _held = _residual_rig(tmp_path, mock_tokenizer, monkeypatch)
    try:
        scheduler = rig.scheduler
        capacity = scheduler._capacity_estimate(
            context_tokens=len(PROMPT_C), max_tokens=3, text_only=True
        )
        assert capacity is not None and capacity.fits_alone
        assert scheduler._fits_later(capacity) is True
        assert scheduler._fits_later(None) is False
    finally:
        release.set()
        rig.close()

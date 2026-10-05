# SPDX-License-Identifier: Apache-2.0
"""A singleton Lightning MTP row hands off to the standard step for a pause.

A decode pause stores the KV of every streamed token and resumes from the
pending ``_next_tokens``, which only the standard step defines. The handoff
waits until at most one committed token is still queued, then leaves the
batch in the standard invariant: the KV holds exactly prompt plus streamed
tokens, and ``_next_tokens`` is the next token an uninterrupted MTP run
streams. The tests drive the real patched mlx-lm ``BatchGenerator`` on the
MTP suite's ``CountingModel``, whose KV stores its input ids verbatim.
"""

from collections import deque
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from test_mlx_lm_mtp_patch import CountingModel

from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_lm_mtp import cache_rollback

PROMPT = [1, 2]
LIMIT = 24


def _generator(model):
    bg.apply()
    cache_rollback.apply()
    return BatchGenerator(
        model,
        sampler=lambda lp: mx.argmax(lp, -1),
        prefill_batch_size=1,
        max_tokens=LIMIT,
    )


def _run(gen, out, limit=200):
    for _ in range(limit):
        _, responses = gen.next()
        for response in responses:
            out.append(response.token)
            if response.finish_reason:
                return response
    raise AssertionError("generation did not finish")


def _uninterrupted():
    gen = _generator(CountingModel())
    try:
        gen.insert([PROMPT], max_tokens=[LIMIT])
        out = []
        _run(gen, out)
        return out
    finally:
        gen.close()


def test_a_pause_handoff_keeps_the_exact_cache_and_greedy_identity_with_mtp():
    expected = _uninterrupted()
    model = CountingModel()
    gen = _generator(model)
    (uid,) = gen.insert([PROMPT], max_tokens=[LIMIT])
    out, results, mtp_steps, requested = [], [], 0, False
    try:
        for _ in range(200):
            batch = gen._generation_batch
            state = getattr(batch, "_omlx_mtp_state", None)
            if state is not None:
                mtp_steps += 1
            # Ask for the pause at a step start with a deep queue.
            requested = requested or (
                len(out) >= 4 and state is not None and len(state.queue) > 1
            )
            if requested:
                queued = len(state.queue)
                result = bg.handoff_singleton_mtp_for_pause(batch)
                results.append((queued, result))
                if result == "handed_off":
                    break
                assert result == "deferred" and queued > 1
            _, responses = gen.next()
            out.extend(r.token for r in responses)
        # Deferred while more than one committed token was queued.
        assert results[0] == (2, "deferred")
        assert results[-1][1] == "handed_off" and results[-1][0] <= 1
        assert mtp_steps > 0  # MTP really ran before the pause
        assert not hasattr(batch, "_omlx_mtp_state")

        # The standard invariant the pause relies on.
        pending = int(batch._next_tokens[0].item())
        cache, tokens = gen.extract_cache([uid])[uid]
        offset = cache[0].offset
        assert offset == len(tokens) == len(PROMPT) + len(out)
        assert tokens == PROMPT + out
        assert cache[0].keys[0, 0, :offset, 0].tolist() == tokens
        out.append(pending)
    finally:
        gen.close()

    # Resume from the stored state on a fresh generator (MTP on again).
    resumed = _generator(CountingModel())
    try:
        resumed.insert(
            [[pending]],
            max_tokens=[LIMIT - len(out)],
            caches=[cache],
            all_tokens=[tokens],
        )
        final = _run(resumed, out)
    finally:
        resumed.close()
    assert final.finish_reason == "length"
    assert out == expected


def test_a_deep_queue_defers_the_handoff_and_a_drained_one_hands_off(monkeypatch):
    lp = mx.zeros((8,))
    calls = []
    monkeypatch.setattr(
        bg,
        "_handoff_mtp_for_late_join",
        lambda batch, state: calls.append(len(state.queue)) or True,
    )
    state = bg._MtpState(uid=7, queue=deque([(1, lp, "draft"), (2, lp, "bonus")]))
    batch = SimpleNamespace(uids=[7], _omlx_mtp_state=state)

    assert bg.handoff_singleton_mtp_for_pause(batch) == "deferred"
    assert calls == []
    state.queue.popleft()
    assert bg.handoff_singleton_mtp_for_pause(batch) == "handed_off"
    assert calls == [1]


def test_no_mtp_state_is_already_standard_and_a_stale_state_fails():
    assert bg.handoff_singleton_mtp_for_pause(SimpleNamespace(uids=[3])) == "standard"
    stale = bg._MtpState(uid=9, queue=deque())
    batch = SimpleNamespace(uids=[3], _omlx_mtp_state=stale)
    assert bg.handoff_singleton_mtp_for_pause(batch) == "failed"


def test_a_handoff_that_cannot_feed_the_next_token_fails(monkeypatch):
    monkeypatch.setattr(bg, "_handoff_mtp_for_late_join", lambda batch, state: False)
    state = bg._MtpState(uid=7, queue=deque())
    batch = SimpleNamespace(uids=[7], _omlx_mtp_state=state)
    assert bg.handoff_singleton_mtp_for_pause(batch) == "failed"

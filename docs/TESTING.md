# Cluster test filesystem isolation

The autouse `cluster_home` fixture gives each test a temporary directory for cluster interpreter shims and SSH files. It preserves explicit shim `home` arguments and leaves `HOME` unchanged so model discovery paths still work. The shim unit tests import the original function directly and provide their own temporary paths or patch `HOME` to verify the real default-path behavior.

The audio model-list smoke test checks that app startup publishes its shim inside the isolated directory. The cluster GET-route smoke test checks that `/ssh-key` creates its key pair there.

# macOS readability theme tests

Run `xcodebuild -project apps/omlx-mac/oMLX.xcodeproj -scheme oMLX -destination 'platform=macOS' -only-testing:oMLXTests/ThemeTests test` to check theme colors. The readability case renders a probe through `.omlxThemed()` with isolated saved preferences, checking disabled and enabled colors in light and dark appearances. It covers the shared theme path used by popovers, not app relaunch or full-screen layout.

# Text-only VLM loading tests

Run `python -m pytest -q tests/test_vlm_vision_fallback.py` to check strict loading and logits with a small quantized DiffusionGemma checkpoint, unchanged loaders for unreadable shards or retained vision, and patch restoration after loading errors. No model download is required.

# Test timing

CI runs all default tests on Python 3.11 for PRs and pushes to `main`. A daily scheduled run, which can also be started manually, covers Python 3.11, 3.12, and 3.13. Each run reports the 50 slowest phases and uploads `test-results.xml` as `test-results-py<version>`. Use `python -m pytest --durations=50 --junitxml=test-results.xml` to collect the same timing data locally. Compare runner queue time separately from test execution.

The automatic Qwen FP16/BF16 decode route has numerical, cache-state and
fallback tests in `tests/test_qwen35_fp16_decode.py`. Run it with
`tests/test_qwen35_gdn_prework.py` to check that the existing BF16 Qwen4 and
speculative routes remain intact. See
[GDN decode prework](experimental/qwen35_fp16_decode.md) for the hardware, geometry
limits and real-model benchmark requirements.

# First-token burst release

Run `python -m pytest -q tests/test_engine_core.py tests/test_output_collector.py`
to check first-chunk delivery across the executor boundary, late admission,
multi-token chunks, output ordering, later burst limits and request cancellation.
Burst decode releases each request's first generated chunk before continuing
with the configured burst policy. This adds one executor hand-off per request;
it does not shorten prefill or bypass parser, stop-string or stream-interval
buffering. Subsequent chunks still follow the selected Burst Decode setting.

For a real-server comparison, use the same model, prompt, output length and
cache state on main and the branch. Measure client-observed first content and
complete-response time separately from producer-side token timestamps, with
balanced (0.1 s) and aggressive (0.2 s) burst settings. Include short replies,
long replies, a second request admitted during decode, and disconnect/recovery.
Report any custom budgets separately from the stock modes.

Cluster process-group tests use the `mock_cluster_ssh` fixture; remote teardown and serve-marker tests retain their own transport assertions. Mock-model engine tests skip explicit GC, while `test_engine_teardown.py` and `test_per_engine_threads.py` retain teardown and reclamation coverage. GLM5 execution tests reuse the eight-layer KDA/DSA fixture with dense and MoE layers; checkpoint-key tests retain the 45-layer configuration. The SDPA memory test retains the 8K/32K length ratio, head dimension 256, and 6:1 GQA ratio with fewer heads. DeepSeek V4.1 direct and converted engine checks run sequentially in one isolated subprocess with separate checkpoint directories.

# Cache cleanup logging tests

Run `python -m pytest -q tests/test_vision_feature_cache.py tests/test_paged_ssd_cache.py -k "cleanup_unlink_failure or corrupt_block_cleanup_logging"` to check failed-delete warnings and cache cleanup state. These cases use the existing cache fixtures to close writer threads.

# Cache-preserving engine teardown tests

Run `python -m pytest -q tests/test_engine_teardown.py tests/test_engine_core.py tests/test_scheduler.py tests/test_paged_ssd_cache.py tests/test_hot_cache.py tests/test_engine_pool.py tests/test_batched_engine.py tests/test_vlm_engine.py` to check the 60-second teardown budget and one progress-gated extension to 120 seconds. Clock tests cover the deadlines; short-budget subprocesses exercise fatal exits. Real writer-thread cases verify primary/draft flushes and in-flight prefix stores survive a saturated queue and can be reused after reload. Async cases cover event-loop responsiveness, cancellation, and leases during another model's unload.

# Claude launcher tests

Run `python -m pytest -q tests/test_cli.py tests/test_integrations.py` to check model selection, saved tier precedence, and the final Claude Code environment. Mixed-context cases verify that the selected initial model is passed through `ANTHROPIC_MODEL` and that the process-wide context and auto-compact limits use the smallest known context window across the initial model and configured tiers. Larger models consequently compact at that shared limit; models without reported limits cannot contribute a limit.

# ModernBERT embedding tests

Run `python -m pytest -q tests/test_modernbert_attention.py tests/test_embedding.py tests/test_mlx_embeddings_compat.py` to check finite padded attention, single-input equivalence, local-window masking, and embedding integration. The attention regression covers fp16, bf16, and fp32 at lengths around the affected SDPA tile boundaries.

# QSA reservation tests

Run `python -m pytest -q tests/test_qwen4_qsa_reserved_capacity.py` to check QSA capacity reservations.

The integration tests cover restored-prefix lengths with boundary snapshots enabled and disabled, the first allocation after cache restoration, and prefill/decode output equivalence using a small Qwen4 model.

Related regression suites are `test_qwen4_qsa_incremental_cache.py`, `test_qwen4_qsa_decode_gather.py`, and `test_prefill_oom_graceful.py`.

For Qwen4 native sparse-GQA prefill measurements, run `python benchmarks/bench_qwen4_qsa_sparse_gqa.py --key-tokens 24576 --query-tokens 1024 --repetitions 30`. The benchmark reports index scoring, top-k selection, the combined native pipeline, every supported main-attention tile, the portable reference, and maximum error. Production groups native query rows into 4,096-row tiles through 32K keys, 2,048-row tiles through 64K, and 1,024-row tiles above 64K; this bounds the FP32 score sheet while amortizing per-tile dispatch.

# Qwen4 verify attention row tests

Run `python -m pytest -q tests/test_qwen4_verify_attention_rows.py` to check that row-exact Lightning MTP verify windows through Qwen4 attention give every row the bits of the serial one-row decode step and leave the same KV and QSA indexer state. The tests build one attention layer at the real Flash-Next shapes with synthetic 6-bit weights. Masked-arm windows (past the 2,048-token QSA budget, rank-three positions) cover 2 to 8 rows at 2,060, 16,382 and 24,000 cached tokens and compare each row's FP32 block scores and token mask; a rollback case accepts one draft and decodes on. Dense windows below the budget include rows on both sides of MLX's one-pass/two-pass vector SDPA switch at 1,024 keys. `OMLX_QWEN4_QSA_MASKED_VERIFY=0` restores the multi-row masked path.

`test_mlx_vlm_qwen4_exp_compat.py::test_qwen4_mtp_one_row_step_is_the_serial_decode_step` checks that a one-row Lightning MTP window (the activation step and depth-0 cycles) runs the serial decode step: equal logits and cache state, no speculative transaction, and a following verify window that rolls back as usual. `OMLX_QWEN4_MTP_ONE_ROW_DECODE=0` keeps the verify forward for those windows.

# Prefill memory accounting tests

`python -m pytest -q tests/test_engine_preflight.py` checks route admission after eviction or reclaim. Both batched wrappers must refresh their cached MLX sample on the owning executor, including when the pool reports that no action was necessary. Controlled memory readings cover newly available headroom, insufficient headroom, and requests that already fit without executor work.

Run `python -m pytest -q tests/test_prefill_transient_tracker.py tests/test_prefill_oom_graceful.py` to check retained versus reclaimed overhead, configured chunk sizes, and abort-cap enforcement. The loop tests run a small initialized MLX model with controlled footprint readings through external and chunked prefill; they do not load a checkpoint.

# Prefix cache completion tests

Run `python -m pytest -q tests/test_scheduler.py tests/test_scheduler_boundary_completion.py tests/test_prefix_cache_gdn_split.py` to check cache-freshness admission and completed boundary recovery. The completion tests use a small initialized Qwen3.5 hybrid model and the real BatchGenerator, then compare restored-prefix logits with a fresh forward pass. They cover embedded snapshots, GDN sidecars, exact SpecPrefill static-prefix sidecars, off-boundary completion, and unknown or inconsistent cache positions.

# Cluster join recovery tests

Run `python -m pytest -q tests/test_cluster_pairing_session.py tests/test_cluster_pairing.py tests/test_cluster_ui_integration.py tests/ui/test_cluster_v2_wizard.py` to check joining, cancellation, and approval. Session tests recreate a manager with the same base path to verify that the original code and cancellation proof survive a restart. They also cover offline cancellation, switching peers while cleanup is pending, rejected requests versus lost responses, and storage failures. Wizard tests check that delayed responses cannot restore a cancelled join and that pending cleanup leaves new pairing controls available.

For a two-Mac smoke test, start isolated servers with separate base paths. Request a join, restart only the joining server, then cancel and retry; the coordinator must remove the original pending request. Repeat with the coordinator offline: cancellation must return `state: idle` with `cleanup_pending: true`, and joining a different reachable Mac must work. Restore the coordinator and poll the join endpoint to verify cleanup. Keep existing approval/cancel race and token-ownership tests in the run; never relax the coordinator's token check to make a stale request disappear.

# DeepSeek V4.1 offline tests

CED scheduler prefill, short suffix positions, full-logit scoring, and DSpark
continuity: `python -m pytest tests/test_deepseek_v41_ced.py tests/test_deepseek_v41_ssd.py -q`.
Replay is approximate; cache restoration is compared with the same chunk boundaries.

Run `python -m pytest -q tests/test_deepseek_v41.py` for the text/vision port. Synthetic weights and small recorded official outputs cover prefill/decode, DSpark, Engram and vision numerics without a checkpoint download or vendored reference implementation. See `tests/fixtures/deepseek_v41_expected.md` for provenance and reference corrections. PyTorch is optional for the independent FP8/FP4 arithmetic checks. Other cases cover BatchGenerator admission, cache restoration, DSML and VLM engine execution.

FP8 activation boundary and layout checks: `python -m pytest tests/test_deepseek_v41_activation.py -q`. These compare fused FP8 and weighted SwiGLU kernels against reference arithmetic across rounding ties, scale transitions, clipping, intermediate casts and floating-point dtypes.

MoE activation reuse and per-expert reference checks: `python -m pytest tests/test_deepseek_v41_moe.py -q`. These cover sorted prefill, decode, independent routed/shared activation policies, and repeated outputs with small quantized weights.

DeepSeek V4.1 Metal arithmetic and sparse-addressing checks: `python -m pytest tests/test_deepseek_v41_kernels.py -q` (no checkpoint required).

Packed attention rounding: `python -m pytest tests/test_deepseek_v41_attention_rounding.py -q`. An independent MLX oracle checks 64-key online maxima, FP32 denominators, BF16 PV probabilities, masking, sink placement, and growing or strided KV across the threadgroup capacity boundary. This does not execute the official CUDA kernels.

Engram storage and prefetch lifecycle: `python -m pytest tests/test_deepseek_v41_offload.py -q`. Modal state, forced-toggle behavior, and save payload: `node --test tests/deepseek_v41_offload_ui.test.cjs`. Both use synthetic fixtures and require no checkpoint download.


### MoE expert residency

`tests/test_deepseek_v41_moe_offload.py` exercises original and converted
V4.1 expert reads, MXFP4/MXFP8 and mixed-bit affine arithmetic, repeated
evictions, sorted routes, load/inference thread separation, Engram coexistence,
draft-weight exclusion, and memory estimates. The load probe rejects whole
expert reads from shared shards and any expert slab read through the Engram
mapping. Further cases check that consumed read buffers are released within the
in-flight byte window and pin the serial LRU order under concurrent reads,
expert-boundary chunking of sorted routes, and the fit-to-budget residency
helper against the admission arithmetic. `tests/test_deepseek_v41_affine_source.py`
covers community `mlx_lm` affine source checkpoints: packed and declared-dense
projections, exact force-dense dequantization, the affine Engram table spec, a
convert round-trip against a direct load, the declared-format resolver, and
offload eligibility. Run alongside `test_deepseek_v41_offload.py`,
`test_moe_expert_offload.py`, and the engine-pool/model-settings suites.
`node tests/moe_expert_offload_ui.test.cjs` checks the actual dashboard
save/reopen payload and speculative-decoding toggle exclusion.

`tests/test_moe_expert_offload.py` also exercises Qwen4-Exp MoE routing with
512 experts, top-k 10, 64 resident slots, shared experts, and repeated
evictions, plus the resident Lightning MTP head (`mtp.*`) and its admission
pricing. `tests/test_moe_offload_compat.py` covers the model-type allowlist,
checkpoint completeness, dense-model exclusion, API/runtime rejection, and
PLE/Engram metadata after expert savings.

# macOS port persistence tests

Run the `AppConfigTests`, `ServerProcessIntegrationTests`, `ServerScreenVMStorageDiffTests`, `AppServicesPathTests`, and `MenubarControllerPortTests` targets with `xcodebuild test`. For process tests, set `TEST_RUNNER_OMLX_INTEGRATION=1`, `TEST_RUNNER_OMLX_PYTHON_OVERRIDE` to a Python interpreter with oMLX dependencies, `TEST_RUNNER_PYTHONPATH` to the repository root, and `TEST_RUNNER_OMLX_BASE_PATH` to a fresh temporary directory so the test host does not start the user's configured server. Leave `TEST_RUNNER_OMLX_DEV_SERVER_SCRIPT` unset to exercise the real Python API and CLI with empty model directories. Setting it to `apps/omlx-mac/Scripts/dev_server.py` instead runs the lightweight process fixture.

The port cases cover offline Apply, recovery from an occupied port, web API saves followed by manual and automatic restarts, native Apply while running, client endpoint synchronization, and persistence across subsequent starts. Each child uses a temporary settings directory and an OS-selected port, and teardown stops the children. Endpoint-only persistence also checks that unrelated settings and corrupt files are preserved.

# Accuracy benchmark worker tests

Run `python -m pytest -q tests/test_eval_worker_pool.py tests/test_eval.py tests/test_accuracy_benchmark.py tests/test_admin_external_accuracy_diagnostics.py tests/test_accuracy_upload.py`. Worker tests cover slot refilling, thinking-mode retries, sequential code scoring without blocking generation, and repeated cancellation during scoring. Real HumanEval, MBPP, and LiveCodeBench subprocess cases verify normal completion, cancellation draining, and temporary-file cleanup with a controlled engine; no model checkpoint is required.

# Profile API exposure tests

Run `python -m pytest -q tests/test_admin_new_profile_expose_as_model.py tests/test_admin_profiles_api.py tests/test_model_settings_profiles.py`. The new-profile tests check toggle bindings, the OFF reset, and request serialization. Existing API tests cover the edit form, persistence, exposed model IDs, and name collisions.

### Lightning MTP with XTC sampling

Run `python -m pytest tests/test_mtp_xtc_sampling.py -q` for request sampler changes, late-joining mixed batches, row removal, and greedy sampling. These tests use a small MLX model and observe the MTP eligibility boundary; they do not execute a trained MTP head.

### Batched DFlash drafter

Run `python -m pytest tests/test_dflash_batched.py tests/test_mlx_lm_mtp_patch.py -q -k "dflash_batched or block_drafter"`. `test_dflash_batched.py` builds a tiny DFlash2 drafter with random weights and checks that rows drafted together match the same rows drafted alone across ring wrap-around, ragged context segments and cohort changes, plus the prefill seed window slicing and block-size clamping. The `block_drafter` cases in `test_mlx_lm_mtp_patch.py` drive the Lightning MTP verify path with a table drafter on the CountingModel harness and require token parity with standard decoding, one context entry per committed position (including late joins) and release of finished rows. Real drafter acceptance and throughput need a Qwen3.5-family VLM checkpoint with its `z-lab` DFlash draft and are measured against the standard batched engine.

# VLM cache boundary tests

Run `python -m pytest -q tests/test_vlm_cache_boundaries.py tests/test_vlm_engine.py tests/test_prefix_cache.py tests/test_paged_cache.py` to check image-aware prefix keys. Boundary cases cover reasoning-dependent template prefixes, final grid token positions, adjacent images, multiple images per turn, block edges, invalid metadata, and isolation when earlier or later images change. These tests use synthetic processor inputs and KV arrays; checkpoint preprocessing and inference comparisons require local models.


### Profile consistency

Run `python -m pytest tests/test_model_settings_profiles.py tests/test_admin_profiles_api.py tests/test_admin_model_settings_template.py -q`. The editor behavior test executes the dashboard JavaScript with Node.js and is skipped if Node.js is unavailable. The cases cover latest-template application, same-name model preservation, renamed and deleted template references, persistence rollback, create-and-apply behavior, and preserving edits during focus refresh. The macOS `ProfileScopeTests` cover source references, stale copies, orphan visibility, and display names. For a manual cross-client check, edit a global profile in the web UI, apply it from the app, and verify the model's effective settings; repeat with an independent same-name model profile and with an unsaved editor open.

Global and model profiles may share display names while retaining separate IDs. Applying a global template reads its latest settings; deleting it preserves model copies as independent profiles. Both editors save and apply new profiles, and focus refresh preserves unsaved edits.

Startup reference repair is covered in `tests/test_model_settings_profiles.py`: missing references are cleared without replacing saved values, originals are backed up under `<base_path>/profile-reference-backup-*`, and subsequent loads do not write again. Retries with unchanged originals reuse the same content-hash backup directory and fill missing files. Mismatched backups prevent repair. Invalid storage, unsupported versions, backup failures, and failed writes must not persist inferred repairs. A rollback failure aborts startup and logs the backup path.

## Qwen tool-call recovery

Run `python -m pytest tests/test_tool_calling.py tests/integration/test_e2e_streaming.py -q`. Final Qwen parsing preserves unknown tool names for client feedback, recovers complete functions missing only the outer envelope close at normal EOF, and reports unrecoverable siblings without a successful stop. Cases cover all three streaming APIs, chunk boundaries, repeated calls, literal tags in arguments, schema validation and length stops. Other parser families and reasoning-channel promotion keep their existing rules.

For a real-server check, request a small `write(content: string)` call with thinking disabled and greedy sampling. Compare the normal result with a request using `stop: ["</tool_call>"]`: the complete function should still arrive once with identical arguments. Then supply an assistant call to an unknown tool followed by a matching tool-error message naming `write`; verify the next model turn uses `write`. Use an isolated port and base path, and do not execute model-supplied file operations during the check.

# Streamed oQ calibration tests

Run `python -m pytest tests/test_oq.py -k TestStreamedCalibration` for streamed calibration. The small BF16 Qwen4 fixture exercises GDN, sparse attention, mmap PLE and the MTP head. It compares imatrix statistics and fused sensitivity with resident collection, verifies cache reuse with and without MTP, and converts and reloads the artifact with its shared PLE scale intact. A small MiniMax decoder fixture also compares dense and MoE collection. These cases replace the separate streaming test modules and need no external checkpoint.

# Fused routed-expert decode tests

Run `python -m pytest -q tests/test_qwen35_moe_routed_decode.py tests/test_qwen35_moe_router.py tests/test_qwen35_moe_gate_up.py` to check the one-token routed-expert kernels. Real `Qwen3_5MoeSparseMoeBlock` instances laid out like Qwen3.8-Flash-Next oQ (quantized routed experts, 8-bit shared expert and shared-expert gate, bf16 router) must match the served body bit for bit, with the shared expert and its gate folded into the two launches: 5-bit (oQ5e) and 4-bit experts at the Flash-Next shape (hidden 2560, intermediate 640, top-k 10), and 5-bit gs32, 6-bit gs128 and 8-bit experts at smaller shapes. A bf16 shared expert stays composed and must match too. Both launches are also run in FP32 against MLX's FP32 mat-vecs (routed and shared gate+up after SwiGLU, the gate row, every routed and shared down row), because BF16 outputs hide one-ulp FP32 differences (a fast-math `exp` in the SwiGLU sigmoid passes most BF16 cases but fails these). The kernels bind a one-expert view of the stacked weights; routing to experts 500+ of 512 checks that the view still reads the stacked buffer, and replacing the expert or shared-expert arrays must rebuild the cached plan. The other cases check that shapes where MLX would pick a different mat-vec partition, 3-bit experts, top-k 8, prefill and verify rows, float16, blocks without the gate+up fusion and a kernel failure all keep the served body. The one-launch router softmax + top-k must return the indices and scores of the softmax and top-k launches for random logits and engineered near-ties (every logit repeated eight times, logits on adjacent bf16 values, two-valued rows), a block whose router rows repeat eight times must route like the served block, and the softmax runs in FP32 against MLX's FP32 softmax (a fast reciprocal or a precise `exp` still routes identically but fails there). The router gemv must return MLX's `x @ W.T` logits bit for bit at 512x2560, 256x2048 and 128x1024, and its FP32 row sums must equal MLX's FP32 gemv on the same values (a `simd_sum` in place of MLX's shuffle-down tree changes only a few BF16 logits but every FP32 sum); shapes where MLX reduces K differently (K >= 16 N, a guarded K tail) keep `nn.Linear`. Where the router has 128 to 4,096 experts, its softmax + top-k run inside the gate+up launch for one to three rows: 512-expert blocks (with a folded or a bf16 shared expert) must still match the served body, and in FP32, with router logits tied in pairs, the launch must return the routing launch's indices and scores and the gate+up rows of the launch fed with them. `tests/test_qwen35_moe_verify_window.py` runs every window size with the folded and with the separate routing launch (`OMLX_QWEN35_MOE_TOPK_FOLD=0`).

# Qwen3.5 fused verifier norm

Run `python -m pytest -q tests/test_qwen35_gdn_norm_gate.py` on a Metal-capable
Mac to compare the fused gated RMS norm with the served SiLU graph. Tests cover
FP16/BF16, varied RMS weights, three epsilon values, every gate encoding, and
fallback when the installed MLX arithmetic is unsupported. The check supports
both released and nightly MLX builds; it does not assume the exponential from
the version number.

# Native Qwen video input

Run `python -m pytest -q tests/test_video.py tests/test_image_utils.py tests/test_vlm_engine.py -k "video or media"` to check video input for Qwen3.5-family checkpoints without weights. The cases cover data-URI extraction and the unchanged rejection for other models, the video processor attached after load (official pixel budget, checkpoint config, frame cap), the prompt-token count against mlx-vlm's real processor both under and over the pixel budget, rejection of undecodable or degenerate clips, the order of video and text in the formatted prompt, per-clip prefix-cache ranges, video feature caching, temporary-file cleanup when preprocessing fails, and the preflight budget. Changes to the video path also need a real-model check with Lightning MTP and the batched DFlash drafter, including a follow-up turn on the same clip that reuses the cached prefix.

# Engine idle timing tests

Run `python -m pytest -q tests/test_engine_pool.py tests/test_active_models_visibility.py` to check that lease completion refreshes the LRU/TTL timestamp and that busy models report zero idle time. The cases cover a request longer than its TTL, release with a pending unload, cancelled release, redundant releases, and models with a held lease, an active request, or a waiting request.

# Incremental prefill commit tests

Run `python -m pytest -q tests/test_scheduler_prefill_progress_commit.py tests/test_settings.py -k "prefill_progress or commit"` to check `scheduler.prefill_progress_commit_blocks` (also `OMLX_PREFILL_PROGRESS_COMMIT_BLOCKS`; 0 = off, the default). The cases drive the real chunked and external prefill loops on a small initialized Qwen3.5 hybrid model (inline and split-GDN snapshots) and a small dense model, against real paged-SSD, boundary-snapshot and prefix stores, and compare restored-prefix logits with a cold forward pass. They cover commits every N blocks, a request aborted mid-prefill whose same-prompt re-issue resumes from the last committed boundary, completion after commits (same stored prefix, no duplicate blocks or leftover snapshot markers), a commit window longer than the restored prefix, TurboQuant windows matching whole-prefix conversion bit for bit, the memory-pressure skip, and the cold retry after a prefill memory failure.

Mocked tests do not replace hardware validation: with a real model, abort a long prompt mid-prefill (for example with a low memory ceiling and two concurrent prompts), re-issue it, and check that the log reports a prefix cache restore at the last committed boundary instead of a full prefill.

# Memory scheduling gate tests

Run `python -m pytest -q tests/test_fork_defaults.py tests/test_build_fork_app.py` to check the `OMLX_DEFAULTS_FILE` layer and the fork app packaging. For each fork key (`scheduler.prefill_progress_commit_blocks`, `scheduler.memory_pressure_park`, `scheduler.memory_abort_single_victim`, `scheduler.media_text_suffix_chunked`, `memory.mlx_cache_limit_gb`, `memory.memory_scheduling`, `server.mid_system_policy`, `server.thinking_retention_guard_tokens`) the tests check that the default applies when `settings.json` lacks the key (or has no file), that a `settings.json` value wins, that an explicit env var wins over both, and that without `OMLX_DEFAULTS_FILE` the built-in defaults stay. A missing or malformed file (bad JSON, not an object, a section that is not an object) logs a warning and falls back; unknown sections or keys are skipped with a warning; a relative path resolves next to the `omlx` package. A save (and `save_cli_overrides`) never writes a default into `settings.json`, a changed value is saved as the user's, a `settings.json` value survives a save, and the `OMLX_MID_SYSTEM_POLICY` / `OMLX_THINKING_RETENTION_GUARD_TOKENS` env overrides are never persisted while a `settings.json` value for those keys is kept. The startup report counts applied defaults and those overridden by `settings.json` or env, and `scheduler_switch_env` exports only the enabled env-read switches (`tests/test_cli.py` checks that `omlx serve` exports them and logs the report). The packaging tests check that the bundle holds `Contents/Resources/omlx-fork-defaults.json` with `FORK_DEFAULTS`, that `LSEnvironment` holds only `OMLX_DEFAULTS_FILE`, that the batched-decode gate is not a default, and that every default loads through `omlx.settings` with no override.

Run `python -m pytest -q tests/test_memory_scheduling_gate.py` to check `memory.memory_scheduling` (env `OMLX_MEMORY_SCHEDULING`; `auto`, `on` or `off`, default `auto`). It gates three behaviours: the never-fits 400 with the capacity endpoint numbers and the `X-oMLX-Max-Context-Tokens` header; the fits-later wait, including the solo admission of a lone head; and the progress-based admission-stall clock. `auto` is on when the prefill memory guard is on and an explicit ceiling is set (`memory_guard_tier: custom` with `memory_guard_custom_ceiling_gb` > 0; a ceiling value left under another tier does not count). `on` forces it on but still needs the guard (with the guard off it stays off and logs a warning). `off` forces it off. The `ProcessMemoryEnforcer` resolves the mode at startup and whenever the mode, the guard, the tier or the ceiling changes (the admin settings apply live), logs `Memory scheduling: on (explicit ceiling 48.0 GB)` or the reason it is off, and propagates the result to every scheduler (`Scheduler._memory_scheduling`, False until an enforcer sets it). The tests cover the resolution matrix, the setting (default, `to_dict`, `from_dict`, env override, validation, the admin GET and POST), and, with it off, upstream behaviour on each path: no capacity estimate, snapshot or header (a cached hint is not served either); a request whose `max_tokens` can never fit passes when its prompt fits now; a fits-alone-but-not-now request gets the old current-usage rejection; a lone head is not admitted solo; the stall clock fails the head 60 s after it first blocked, whatever else progresses, with the old message and metadata (no `no_progress_seconds`), also for store-cache backpressure; with the park feature on, a hold it created (a parked or decode-paused request running solo or waiting) keeps the progress clock and its 600 s bound whatever the mode, because the head waits for work held back on purpose (park off, no such hold exists); an external-loop prefill is not reported as solo to the enforcer; and the capacity endpoint returns `available: false` with `memory_scheduling: "off"` and the reason, while unknown and unloaded models keep their 404 and 409. `tests/test_memory_external_prefill.py` also drives the real enforcer with the fork flags and memory scheduling off: the lone external prefill at hard pressure gets upstream's abort-all. The suites below that test these behaviours set the mode on explicitly (`scheduler._memory_scheduling = True`, or `memory_scheduling="on"` on the real enforcer).

Run `python -m pytest -q tests/test_memory_scheduling_admin_state.py` to check memory scheduling across admin settings saves with the `OMLX_DEFAULTS_FILE` layer. The cases load the real settings over a defaults file (`memory_scheduling: "on"`), build the live enforcer from them as the server does, and drive the real admin routes. A save of `max_concurrent_requests`, `chunked_prefill` or `memory_guard_tier` keeps every fork default on the live settings object (and out of `settings.json`, with a reload seeing them again), keeps the mode on and propagated, and logs no `Memory scheduling:` line. Loading the settings page (`GET /admin/api/global-settings`, which builds one tier-preview enforcer per tier) logs none either: only an enforcer that drives an engine pool logs its state. An explicit `memory_scheduling` POST, including one that repeats the current mode, re-resolves, logs the state and propagates it to the schedulers. The GET reports the live state as `memory.memory_scheduling_effective` (`enabled` and `reason`, `null` without an enforcer), the POST response carries it, and the factory defaults response does not.

# Never-fits gate tests

Run `python -m pytest -q tests/test_scheduler_capacity_gate.py tests/test_scheduler_capacity_vlm.py tests/test_scheduler_deferred_admission.py tests/test_scheduler_solo_admission.py tests/test_server_context_capacity.py` to check the feasibility gate. A request whose weights + exact KV of prompt plus `max_tokens` + prefill transient exceed the capacity line (the tighter of the admission line and the prefill safety cap) is rejected with `ContextExceedsCapacityError` (HTTP 400, `omlx_code: context_exceeds_capacity`, the byte and token fields, `retryable: false`) whatever the current usage. One that fits alone but not now is not rejected: the route preflight lets it queue while other work is admitted (queued requests hold no memory and do not count), and the in-stream admission keeps it at the head (nothing overtakes it) until the admitted rows free memory; both checks share one predicate. With nothing admitted, a head that fits alone but whose peak priced on the current usage (which also holds memory no admitted request owns, such as queued requests' vision embeddings) lands between the admission line and the solo line (the ceiling and prefill safety cap minus 2%) runs alone, as a parked head does: `Admitting request <id> alone (solo)`, nothing is admitted beside it, and the route preflight lets it queue. Over the solo line, or with the park feature off, the older rejection applies. The solo-admission cases drive the real scheduler loop with the admission line set by a hard watermark under the ceiling. A deferred request releases its restored prefix and its block-table refs while it waits, and its re-admission repeats the lookup. The block-table release can wait behind an in-flight prefill commit; it is keyed to the deferred attempt's table, so it never frees a table the request got since. The deferred head is not retried (and its prefix not restored again) on every step: it retries once an admitted row left (finished, aborted or paused), free memory rose by more than 2% of the admission line, or its backoff expired (0.25 s doubling per deferral, capped at 4 s); it keeps the head-of-line hold meanwhile, also while its release still waits behind a commit. Both checks price the same output budget: the route preflight uses the resolved `max_tokens` (request, model setting or server default; the engine default when an engine is called directly). With nothing admitted, or when the capacity cannot be estimated (no memory limit, unknown weights or KV dimensions), the older current-usage rejection applies; unknown weights are never cached and never produce a never-fits rejection. The server cases cover the 400 body on API and non-API routes, `GET /v1/omlx/capacity` (fields, 404/409, auth) and the `X-oMLX-Max-Context-Tokens` header on streaming and non-streaming chat completions. `tests/test_scheduler_capacity_vlm.py` builds a tiny real mlx-vlm Qwen3.5 model (vision tower plus hybrid language model) behind the real `VLMModelAdapter`: the adapter keeps the model in `_vlm_model`, which `nn.Module.parameters()` skips, so the weights are summed from the unwrapped model, each array once. It checks the capacity of that model and of the text model on the batched engine, that `kv_bytes_per_token` prices the TurboQuant prefill peak (full-width KV plus the quantized copy, the last full-attention layer kept full width when there is more than one), that quantized weights count their scales and biases, and that `max_tokens` 2,000,000 reaches the gate unclamped (no route clamps `max_tokens` to the context window) and is rejected as never-fits. These cases run with memory scheduling on (see the memory scheduling gate tests above); off, there is no capacity estimate and the older current-usage rejection applies.

Mocked tests do not replace hardware validation: with a real model, send a prompt larger than `max_viable_context_tokens` and check for an immediate 400 with the fields, then send two prompts that each fit alone but not together and check that the second waits and completes without an error.

# Media text-suffix tests

Run `python -m pytest -q tests/test_scheduler_media_text_suffix.py` to check `OMLX_MEDIA_TEXT_SUFFIX_CHUNKED` (off by default; `omlx serve` exports it when `scheduler.media_text_suffix_chunked` is true). With it on, a media request whose cached prefix reaches the token after its last image, video or audio placeholder (`Request.vlm_media_end`, computed by the VLM engine from the processor's token ids) prefills the uncached text suffix through the chunked text path: decode fairness interleaves decode steps and the prefill can be parked. The route also needs the captured mRoPE delta, no atomic media prefix, no SpecPrefill, more than one token left, and the proof that every suffix position equals `k + delta` on all three planes; otherwise the external media path runs. The cases drive the real scheduler and `VLMModelAdapter` on a small Qwen3.5 hybrid whose output depends on every mRoPE plane, against real paged-SSD stores. They compare the chunked suffix positions with the external slice, greedy tokens, final cache and committed block hashes with the flag on and off, and cover the fallbacks, a decode row advancing between suffix chunks, a park and resume from the committed blocks, abort, an OOM requeue that retries cold through the external path, and teardown.

Mocked tests do not replace hardware validation: with a real Qwen VL model, send a follow-up turn after an image turn and check the log for the text-suffix route, then compare its greedy output with the flag off.

# Memory-pressure park tests

Run `python -m pytest -q tests/test_scheduler_park.py tests/test_scheduler_park_resume_priority.py tests/test_process_memory_enforcer.py tests/test_settings.py -k "park or Park"` to check `scheduler.memory_pressure_park` (also `OMLX_MEMORY_PRESSURE_PARK`; off by default), `scheduler.max_parks_per_request` (`OMLX_MAX_PARKS_PER_REQUEST`, default 3) and `scheduler.park_resume_ok_polls` (`OMLX_PARK_RESUME_OK_POLLS`, default 3). With the setting on, hard (non-emergency) memory pressure parks the most recent chunked-prefill request instead of aborting: it is re-queued without an error, keeps its stream, and resumes in arrival order once the estimated free memory under the admission line covers its need plus a 2% margin (a paused decode: the KV of its final length plus one decode step; a parked prefill: the KV of prompt plus max_tokens plus the prefill transient; KV that running prefills still have to allocate does not count as free), restoring the blocks that `prefill_progress_commit_blocks` committed (without it the park resumes cold). The oldest admitted request is protected, parked blocks are pinned against eviction until the resume, and the legacy abort remains for emergency pressure, nothing parkable, an exhausted park budget, or external-loop (non-chunked, VLM) and SpecPrefill prefills, which cannot be parked between steps. While the parked head waits, nothing newer is admitted, so a smaller request cannot take the headroom the head waits for; once the head fits, newer requests are admitted behind it. The head never resumes on hard pressure; the stale-monitor escape still releases it, and with nothing admitted it resumes only when it fits (see the resume-thrash tests below). When no estimate is available (no admission line or no model dimensions), the older rule applies: enough consecutive ok pressure polls, or the max-hold escape. The scheduler cases run on a small Qwen3.5 hybrid model and compare a resumed request's output with an uninterrupted run; its prompts must stay below the tiny vocabulary of 32 ids, or the embedding read is undefined and the output is nondeterministic.

Run `python -m pytest -q tests/test_memory_resume_thrash.py` to check that a paused request does not thrash. Beside admitted work a parked head resumes only when current usage plus in-flight prefill reserve plus its need stays under the admission line minus 2% and under the hard watermark minus 5% (`PARK_RESUME_HARD_MARGIN_FRACTION`); the enforcer pauses at the watermark, so resuming right up to it pauses the request again at the next poll. A head parked or paused twice within 60 s (`PARK_THRASH_PAUSES`, `PARK_THRASH_WINDOW_S`) waits, even when it fits, until an admitted row leaves (finishes, aborts or pauses) or free memory rises by more than 5% of the resume line. With nothing admitted, a head that fits resumes; one that fits only under the ceiling (the scheduler hard limit and the prefill safety cap, minus 2%), one that keeps being paused while alone, or one that waited `PARK_MAX_HOLD_S` resumes as the solo request: nothing else is admitted while it runs, and at hard pressure below the emergency threshold the enforcer neither pauses nor aborts it (only when every other loaded engine is idle). The emergency brake beyond the ceiling still aborts it. Decode pauses have their own budget, `scheduler.max_decode_pauses_per_request` (`OMLX_MAX_DECODE_PAUSES_PER_REQUEST`, default 8), apart from the prefill park budget: a decode pause keeps every generated token exactly, while the abort that follows an exhausted budget loses them all. The real-rig cases wire the real enforcer to the real scheduler and script the memory readings of the C7 run (a lone request decoding over the hard watermark; three requests one at a time; two rows where the newer one is paused twice): every request completes with its uninterrupted greedy output and no abort.

Run `python -m pytest -q tests/test_memory_external_prefill.py` to check memory pressure while a prompt prefills in the external loop (a media prompt whose images are not all cached; it cannot be parked). A request in that loop with nothing else admitted runs alone: at hard pressure below the emergency threshold the enforcer neither pauses nor aborts it. With a decoding row beside it, the decoding row is paused instead. When the single-victim selection finds no live request (queued requests hold no memory), the non-emergency path aborts nothing; before, it fell back to aborting every request. At emergency pressure (beyond the ceiling) the enforcer first pauses the largest pausable decoding row when freeing its estimate brings usage back under the ceiling, and waits for that pause at most two polls (`_EMERGENCY_PAUSE_MAX_POLLS`) and only while usage does not grow; abort-all stays the last resort when nothing is pausable, no pause gets under the ceiling, or the pause is declined or too slow. With the park feature on, a media prefill beside admitted work whose estimated peak (current usage, exact KV and transient) crosses the resume line (the hard watermark minus 5%) waits at the head until an admitted row leaves; it does not pause a decoding row for room, because the paused row would be parked ahead of it and resume first. The external-loop tests block the scheduler's own thread inside the external prefill so the real enforcer polls while it runs, as in production. The lone external prefill counts as solo only with memory scheduling on; these tests set it on.

Run `python -m pytest -q tests/test_scheduler_admission_progress.py` to check that a queued request waits while memory is scheduled on purpose. The admission-stall clock measures time without progress: a token emitted, a prefill chunk processed, an external prefill finished, a request admitted or a store-cache write drained restarts it. Without progress the head fails after 60 s; while a solo request runs, a fits-later head backs off, or a parked or decode-paused request waits, it fails only after `scheduler.admission_no_progress_timeout_s` (`OMLX_ADMISSION_NO_PROGRESS_TIMEOUT_S`, default 600 s) without progress, so a genuine deadlock still surfaces with `memory_admission_stalled` and `no_progress_seconds`. Store-cache backpressure is not a failure while the writer drains. The scheduler logs `Admission waiting for memory: <reason>; <n> request(s) progressing` every 30 s per head. The real-rig cases drive the real enforcer and scheduler on a controlled clock (2 s per step): three requests that fit one at a time complete in sequence (the old clock failed the second after 60 s while the first still decoded), and a wedged decode still fails the queued request. These cases run with memory scheduling on; off, the clock is upstream's fixed 60 s from when the head first blocked, unless a park hold is active. A real-rig case runs park on and memory scheduling off: a request paused twice while alone resumes as the solo request, and the head queued behind it waits while the solo request decodes and then completes (the fixed clock failed it after 60 s, as in C7).

Run `python -m pytest -q tests/test_scheduler_forced_decode_pause.py` to check the test-only knob `OMLX_DEBUG_FORCE_DECODE_PAUSE_AT` (comma-separated output-token offsets, off by default, never for production). When set, the scheduler asks for a decode pause of each decoding row as its output reaches every offset, through the normal decode-pause path (the pending token is emitted, the exact prefix stored, the row removed and re-queued parked), and resumes it at once without the memory hold. A B=1 run with forced pauses (a block edge, mid-block, back to back) gives the same tokens, streamed tokens and final text as a B=1 run without; unset, nothing changes. On a real model it isolates the pause from batch-size numerics (the C7-identity bench scenario).

Mocked tests do not replace hardware validation: with a real model, run three concurrent long prompts under a ceiling too low for all of them and check that each completes with the park/resume log lines and a cached count close to the last commit, and that no client sees a memory error.

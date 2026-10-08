# oMLX Fork app (no Xcode)

`build_fork_app.sh` repackages the official `/Applications/oMLX.app` as
**oMLX Fork.app** plus a drag-to-Applications DMG. The official app is only a
read-only donor: its Swift binary, Python runtime and compiled kernels are
reused, and `Contents/Resources/omlx` receives this repo's `omlx/` package.

Output goes to `packaging/_fork_dist/` (git-ignored):
`oMLX Fork.app` and `oMLX-Fork-<version>.dmg` (for example `0.7.0-fork.1`).

## Build

```bash
packaging/build_fork_app.sh                 # app + DMG, ~1.5 min
packaging/build_fork_app.sh --no-dmg        # app only
packaging/build_fork_app.sh --fork-rev 2    # version 0.7.0-fork.2
.venv/bin/python -m pytest tests/test_build_fork_app.py -q
```

The build refuses to run unless the donor is version `0.7.0` build `2987`
(`--expect-version`, `--expect-build` override) and its site-packages pins
(mlx, mlx-metal, mlx-lm, mlx-vlm, dflash-mlx, transformers, fastapi) equal the
fork venv. A mismatch lists every difference and copies nothing.

What it changes in the copy:

- `Resources/omlx`: fork sources overlaid with the same filter as
  `apps/omlx-mac/Scripts/build.sh`. Donor-only build products (compiled custom
  kernels, `_build_info.py`, `_engine_commits.json`, `tailwindcss`) are kept.
  Stale `.pyc` of changed modules are removed.
- `Info.plist`: bundle id `app.omlx.fork`, name `oMLX Fork`, version
  `0.7.0-fork.N` (the Swift version parser reads only the numeric prefix and
  the suffix has no dev/alpha/beta/rc marker), and `LSEnvironment` with a
  single variable, `OMLX_DEFAULTS_FILE=omlx-fork-defaults.json`.
- `Resources/omlx-fork-defaults.json`: the fork defaults (see
  [Fork defaults](#fork-defaults)). They are defaults, not fixed values: every
  one can be changed from `~/.omlx/settings.json` (or the admin UI where it is
  exposed) without a rebuild.
- **Self-update disabled**: the release URL string inside the Swift binary
  (`api.github.com` -> `no-update.test`, same length) is patched, so the
  updater can never find a release to download or swap in.
  "Check for Updates" reports an error instead.
- Quarantine stripped, then ad-hoc re-signed inside-out with the donor's
  hardened runtime and entitlements (`disable-library-validation` is what lets
  the hardened runtime load the ad-hoc dylibs).
- `Resources/FORK_BUILD.json` records donor/fork versions, the fork commit,
  the `LSEnvironment` and the defaults.

Gatekeeper (`spctl -a`) rejects ad-hoc apps; that is expected for local use.
Installing from the DMG is fine because a locally built DMG has no quarantine.

## Fork defaults

`omlx/settings.py` reads the file that `OMLX_DEFAULTS_FILE` names (a relative
path resolves next to the `omlx` package, so `Contents/Resources` wherever the
app is installed). It has the `settings.json` shape and sits under it:

**explicit env var > `settings.json` > fork defaults file > built-in defaults**

A default applies only when `settings.json` does not contain that key. Values
taken from the defaults file are never written to `settings.json`: a save (the
admin UI, the app's own rewrites, a CLI flag) leaves the key out unless you
changed it. Without `OMLX_DEFAULTS_FILE` (the official app, `omlx serve`)
nothing changes. LaunchServices applies `LSEnvironment` only when the app is
opened from Finder, Dock, `open` or a login item, not when the binary is run
directly.

| Key | Fork default | Built-in | What it does |
| --- | --- | --- | --- |
| `scheduler.prefill_progress_commit_blocks` | `4` | `0` | D1: commit prefill progress to the prefix cache every 4 blocks, so an aborted or parked prompt resumes there. |
| `scheduler.memory_pressure_park` | `true` | `false` | Hard pressure parks the newest prefill (or pauses a decode) instead of aborting. |
| `scheduler.memory_abort_single_victim` | `true` | `false` | Hard pressure with nothing parkable aborts only the largest request (emergency and unload still abort all). |
| `scheduler.media_text_suffix_chunked` | `true` | `false` | A media prompt whose images are all cached prefills its text suffix chunked: decode keeps running and the prefill is parkable. |
| `memory.mlx_cache_limit_gb` | `16` | `0` (unbounded) | `mx.set_cache_limit` at startup; keeps MLX's buffer pool from growing to 65-73 GB during MTP batch decode. |
| `memory.memory_scheduling` | `"on"` | `"auto"` | Never-fits 400, fits-later wait and the progress-based stall clock, whatever the guard tier. |
| `server.mid_system_policy` | `"inline_note"` | unset | pi's mid-transcript system messages render in place as `<system-update>` user messages instead of being folded into the system head, so the prefix cache survives. A per-model setting overrides it. |
| `server.thinking_retention_guard_tokens` | `20000` | unset (off) | pi's image-marker and system-update messages no longer strip the agent loop's thinking; retained thinking is capped at about 20000 tokens (unbounded `preserve_thinking` corrupted output at about 50k). A per-model setting overrides it. |

The batched-decode gate (`scheduler.max_batched_decode_context_tokens`) is no
longer a fork default: memory pressure pauses a decoding row into the SSD
cache and resumes it from its exact prefix, so batches are left to the memory
enforcer.

The two `scheduler.*` switches that runtime code reads from the environment
(`memory_abort_single_victim`, `media_text_suffix_chunked`) are exported as
`OMLX_MEMORY_ABORT_SINGLE_VICTIM` / `OMLX_MEDIA_TEXT_SUFFIX_CHUNKED` by
`omlx serve` when enabled.

These two env vars and `OMLX_MEMORY_PRESSURE_PARK` share one truthy set
(`omlx/utils/env_flags.py`): `true`, `1`, `yes` or `on`, case and surrounding
spaces ignored; any other value is off.

The server log shows one startup line, for example
`Fork defaults: 7 applied, 1 overridden by settings.json (memory.memory_scheduling), 0 overridden by env`.
A missing or malformed defaults file logs a warning and every setting falls
back to `settings.json` and the built-in defaults. A value its section rejects
(for example `memory.soft_threshold: "high"`) is skipped with a warning and the
startup line counts it as rejected; `settings.json` and the other defaults
still load. A value its section coerces (a non-boolean switch reads as off) is
applied as coerced and named in a warning.

### Override a fork default

Quit oMLX Fork first: the app rewrites `settings.json` while it runs, and an
edit made meanwhile can be lost. Then add the key under its section in
`~/.omlx/settings.json` and relaunch. For example, to turn memory scheduling
back to `auto` and re-enable the batched-decode gate:

```json
{
  "memory": {
    "memory_scheduling": "auto"
  },
  "scheduler": {
    "max_batched_decode_context_tokens": 200000,
    "batched_decode_hold_max_s": 120.0
  }
}
```

Merge these keys into the existing sections; do not replace the file. Settings
shown in the admin UI (for example memory scheduling) can be changed there
instead, and the change is saved as your value.

To go back to a fork default, quit the app, delete the key from
`settings.json`, and relaunch.

Older fork builds put these values in `LSEnvironment`, and any save then wrote
them into `settings.json`. Those copies now count as your values and win over
the fork defaults; the startup line lists them as "overridden by
settings.json". Delete a key to let the fork default apply again. In
particular, a `scheduler.max_batched_decode_context_tokens` of `200000` left by
an older build keeps the gate on.

## Install

1. Quit the official oMLX (menu bar item, Quit) and stop any fork launchd
   service on port 8001.
2. Open the DMG and drag **oMLX Fork** to Applications (or run
   `ditto "packaging/_fork_dist/oMLX Fork.app" "/Applications/oMLX Fork.app"`).
3. Launch it from Finder (not the terminal binary, so `LSEnvironment` applies).
4. Check the server log for the `Fork defaults:` startup line.

New bundle id means macOS asks again for permissions. Cache on a removable
volume (for example `/Volumes/ExternalSSD...`) triggers a TCC prompt for
"removable volumes" on first access; allow it. The old grant belonged to the
official signature and does not transfer.

## Do not run both apps at once

Both share `~/Library/Application Support/oMLX` (control socket
`control.sock`, `update-prefs.json`, `base-path`, logs), `~/.omlx`
(settings, `bin/omlx` and `bin/omlx-cluster-python` shims, which are
re-pointed at whichever app launched last), port 8001, the `omlxapp://` URL
scheme, and the Keychain service `app.omlx.hf-upload`. Quit one before
starting the other.

What a different bundle id resets: UserDefaults (menu bar and appearance
prefs), TCC permissions, and notification settings. `settings.json` keys the
Swift app does not know (`scheduler.*`) are preserved verbatim on every save.

## Revert

1. Quit oMLX Fork and delete `/Applications/oMLX Fork.app`.
2. Launch the official `/Applications/oMLX.app`; it republishes the
   `~/.omlx/bin` shims to itself. Nothing else was modified.

The donor is never written to. Check with
`codesign --verify --deep --strict /Applications/oMLX.app`.

## Rebuild after rebasing the fork

1. Rebase `feat/unattended-scheduling` and refresh the venv.
2. If the official app was updated, expect the donor check to fail: bump
   `--expect-version`/`--expect-build` only after confirming the new donor's
   pins still equal the venv and the release URL patch still applies (the
   build fails loudly if the URL string is gone).
3. Bump the suffix: `packaging/build_fork_app.sh --fork-rev N`.
4. Reinstall as above.

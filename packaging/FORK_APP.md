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
  the suffix has no dev/alpha/beta/rc marker), and `LSEnvironment`.
- **D1, park, a bounded MLX buffer cache and serialized long-context decode ON** through `LSEnvironment`:
  `OMLX_PREFILL_PROGRESS_COMMIT_BLOCKS=4`, `OMLX_MEMORY_PRESSURE_PARK=true`,
  `OMLX_MLX_CACHE_LIMIT_GB=16` (`mx.set_cache_limit`, applied at server startup;
  keeps the enforcer's pool from growing to 65-73 GB during MTP batch decode.
  Upstream default is 0 = unbounded),
  `OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS=200000` (a request whose context plus
  the running rows' would exceed this waits and decodes alone on the fused
  single-row kernels; 45k + 130k recovered live, 2 x 130k did not. Upstream
  default is 0 = off).
  The Swift app passes its own process environment to the server
  (`PythonRuntime.makeEnvironment`) and `omlx/settings.py` applies these
  variables after `settings.json`. Upstream defaults in the repo stay off.
  LaunchServices applies `LSEnvironment` only when the app is opened from
  Finder, Dock, `open` or a login item, not when the binary is run directly.
- **Self-update disabled**: the release URL string inside the Swift binary
  (`api.github.com` -> `no-update.test`, same length) is patched, so the
  updater can never find a release to download or swap in.
  "Check for Updates" reports an error instead.
- Quarantine stripped, then ad-hoc re-signed inside-out with the donor's
  hardened runtime and entitlements (`disable-library-validation` is what lets
  the hardened runtime load the ad-hoc dylibs).
- `Resources/FORK_BUILD.json` records donor/fork versions and the fork commit.

Gatekeeper (`spctl -a`) rejects ad-hoc apps; that is expected for local use.
Installing from the DMG is fine because a locally built DMG has no quarantine.

## Install

1. Quit the official oMLX (menu bar item, Quit) and stop any fork launchd
   service on port 8001.
2. Open the DMG and drag **oMLX Fork** to Applications (or run
   `ditto "packaging/_fork_dist/oMLX Fork.app" "/Applications/oMLX Fork.app"`).
3. Launch it from Finder (not the terminal binary, so `LSEnvironment` applies).
4. Check the server log for the D1 and park settings.

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

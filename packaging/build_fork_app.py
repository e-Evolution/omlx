#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repackage the official oMLX.app as "oMLX Fork.app" without Xcode.

The official app is used as a donor: its Swift binary, the venvstacks Python
runtime and the compiled custom kernels are reused as-is, while the Python
server package (``Contents/Resources/omlx``) is replaced by this repository's
``omlx/`` so the fork's scheduler changes ship inside the app.

Run through ``packaging/build_fork_app.sh`` (uses the repo venv). See
``packaging/FORK_APP.md`` for install, revert and rebase instructions.

Deliberately *not* ``packaging.build_fork_app``: the ``packaging/`` directory
has no ``__init__.py`` and must stay importable-by-path only, otherwise it
shadows the PyPI ``packaging`` distribution.
"""

from __future__ import annotations

import argparse
import datetime
import filecmp
import importlib.metadata
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent

PIN_PACKAGES = (
    "mlx",
    "mlx-metal",
    "mlx-lm",
    "mlx-vlm",
    "dflash-mlx",
    "transformers",
    "fastapi",
)

# Turned on for the fork app only, through Info.plist ``LSEnvironment``. The
# Swift app builds the server environment from its own process environment
# (PythonRuntime.makeEnvironment) and omlx/settings.py applies these variables
# after settings.json, so the upstream defaults in the repo stay off.
FORK_ENV = {
    "OMLX_PREFILL_PROGRESS_COMMIT_BLOCKS": "4",
    "OMLX_MEMORY_PRESSURE_PARK": "true",
}

# The Swift updater polls this URL (ReleasesChecker.releasesURL) and can swap
# the running bundle with the official release. Both hosts have 14 characters,
# so the binary is patched in place; the signature is redone afterwards.
UPDATE_URL_OLD = b"https://api.github.com/repos/jundot/omlx/releases"
UPDATE_URL_NEW = b"https://no-update.test/repos/jundot/omlx/releases"

LIB_SUFFIXES = (".so", ".dylib")
SKIP_BINARY_SUFFIXES = (".so", ".dylib", ".metallib")
MACHO_MAGICS = {
    bytes.fromhex(h)
    for h in ("feedface", "feedfacf", "cefaedfe", "cffaedfe", "cafebabe", "bebafeca")
}
FORK_SUFFIX_RE = re.compile(r"-fork\.\d+$")

Runner = Callable[..., "subprocess.CompletedProcess[bytes]"]


class DonorMismatch(RuntimeError):
    """The donor bundle is not the build this script was written against."""


class BuildError(RuntimeError):
    """An external tool failed."""


@dataclass
class Options:
    donor: Path = Path("/Applications/oMLX.app")
    out: Path = REPO_ROOT / "packaging" / "_fork_dist"
    name: str = "oMLX Fork"
    bundle_id: str = "app.omlx.fork"
    package: Path = REPO_ROOT / "omlx"
    expect_version: str = "0.7.0"
    expect_build: str = "2987"
    fork_rev: int = 1
    sign: bool = True
    dmg: bool = True
    env: Mapping[str, str] = field(default_factory=lambda: dict(FORK_ENV))


def default_runner(cmd, **kwargs):
    kwargs.setdefault("capture_output", True)
    return subprocess.run([str(c) for c in cmd], **kwargs)


def _check(proc, what: str):
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode(errors="replace").strip()
        raise BuildError(f"{what} failed (exit {proc.returncode}): {err}")
    return proc


# ---------------------------------------------------------------- validation


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def donor_site_packages(donor: Path) -> Path | None:
    base = donor / "Contents" / "Resources" / "Python" / "framework-mlx-base" / "lib"
    found = sorted(base.glob("python3*/site-packages")) if base.is_dir() else []
    return found[0] if found else None


def read_donor_pins(donor: Path) -> dict[str, str]:
    sp = donor_site_packages(donor)
    pins: dict[str, str] = {}
    if sp is None:
        return pins
    wanted = set(PIN_PACKAGES)
    for dist in sp.glob("*.dist-info"):
        name, _, version = dist.name[: -len(".dist-info")].partition("-")
        name = _norm(name)
        if name in wanted:
            pins[name] = version
    return pins


def read_venv_pins(names=PIN_PACKAGES) -> dict[str, str]:
    return {n: importlib.metadata.version(n) for n in names}


def read_info(donor: Path) -> dict:
    with open(donor / "Contents" / "Info.plist", "rb") as fh:
        return plistlib.load(fh)


def validate_donor(
    donor: Path,
    expect_version: str,
    expect_build: str,
    venv_pins: Mapping[str, str],
) -> None:
    """Fail loudly (one error listing every problem) if the donor is unexpected."""
    problems: list[str] = []
    if not (donor / "Contents" / "Info.plist").is_file():
        raise DonorMismatch(f"{donor} is not an app bundle (no Contents/Info.plist)")
    info = read_info(donor)
    version = info.get("CFBundleShortVersionString")
    build = info.get("CFBundleVersion")
    if version != expect_version:
        problems.append(f"donor version is {version!r}, expected {expect_version!r}")
    if build != expect_build:
        problems.append(f"donor build is {build!r}, expected {expect_build!r}")
    sparkle = sorted(k for k in info if k.startswith("SU"))
    if sparkle:
        problems.append(f"donor has Sparkle keys {sparkle} (SU*); update handling differs")
    donor_pins = read_donor_pins(donor)
    for pkg in PIN_PACKAGES:
        want = venv_pins.get(pkg)
        have = donor_pins.get(pkg)
        if have is None:
            problems.append(f"{pkg}: not found in the donor site-packages")
        elif want != have:
            problems.append(f"{pkg}: donor has {have}, fork venv has {want}")
    if problems:
        raise DonorMismatch("donor validation failed:\n  - " + "\n  - ".join(problems))


# ------------------------------------------------------------------- overlay


def _excluded_dir(rel_parts: tuple[str, ...]) -> bool:
    name = rel_parts[-1]
    if name in ("__pycache__", "tests", ".git"):
        return True
    return len(rel_parts) == 3 and rel_parts[0] == "custom_kernels" and name == "csrc"


def _drop_stale_pyc(dest_file: Path) -> int:
    cache = dest_file.parent / "__pycache__"
    if dest_file.suffix != ".py" or not cache.is_dir():
        return 0
    removed = 0
    for pyc in cache.glob(f"{dest_file.stem}.*.pyc"):
        pyc.unlink()
        removed += 1
    return removed


def overlay_package(src: Path, dest: Path) -> dict:
    """Copy the fork package over the donor's ``Resources/omlx``.

    Mirrors ``apps/omlx-mac/Scripts/build.sh`` (excludes ``__pycache__``,
    ``*.pyc``, ``tests``, ``.git``, ``custom_kernels/*/csrc``) and never
    deletes donor-only files: the donor carries build products that are not in
    the repo (``_build_info.py``, ``_engine_commits.json``, the compiled
    custom kernels, ``tailwindcss``). Compiled files from the repo are never
    copied, so the donor's kernels always win.
    """
    stats = {"added": 0, "changed": 0, "pyc_removed": 0, "donor_only_py": []}
    seen: set[Path] = set()
    for root, dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        dirs[:] = sorted(
            d for d in dirs if not _excluded_dir((*rel_root.parts, d))
        )
        for fname in sorted(files):
            if fname.endswith(".pyc") or fname.endswith(SKIP_BINARY_SUFFIXES):
                continue
            rel = rel_root / fname
            seen.add(rel)
            s, d = src / rel, dest / rel
            if s.is_symlink():
                if d.is_symlink() or d.exists():
                    d.unlink()
                d.parent.mkdir(parents=True, exist_ok=True)
                d.symlink_to(os.readlink(s))
                stats["added"] += 1
                continue
            existed = d.is_file()
            if existed and filecmp.cmp(s, d, shallow=False):
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            if d.exists() or d.is_symlink():
                d.unlink()
            shutil.copy2(s, d)
            stats["added" if not existed else "changed"] += 1
            stats["pyc_removed"] += _drop_stale_pyc(d)
    for root, dirs, files in os.walk(dest):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            if fname.endswith(".py"):
                rel = (Path(root) / fname).relative_to(dest)
                if rel not in seen:
                    stats["donor_only_py"].append(str(rel))
    stats["donor_only_py"].sort()
    return stats


# --------------------------------------------------------------------- plist


def fork_version(base: str, fork_rev: int) -> str:
    """``0.7.0`` -> ``0.7.0-fork.1``.

    Safe for ReleasesChecker.parseVersion: the numeric components are read up
    to the first character outside ``[0-9.]``, and the suffix contains none of
    the dev/alpha/beta/rc markers, so the phase stays "final".
    """
    return f"{FORK_SUFFIX_RE.sub('', base)}-fork.{fork_rev}"


def patch_info_plist(
    plist_path: Path,
    *,
    name: str,
    bundle_id: str,
    fork_rev: int,
    env: Mapping[str, str],
) -> dict:
    with open(plist_path, "rb") as fh:
        data = plistlib.load(fh)
    data["CFBundleIdentifier"] = bundle_id
    data["CFBundleName"] = name
    data["CFBundleDisplayName"] = name
    data["CFBundleShortVersionString"] = fork_version(
        data["CFBundleShortVersionString"], fork_rev
    )
    # CFBundleExecutable stays "oMLX": the Swift binary is not renamed.
    if env:
        data["LSEnvironment"] = {**data.get("LSEnvironment", {}), **env}
    with open(plist_path, "wb") as fh:
        plistlib.dump(data, fh)
    return data


# ----------------------------------------------------------- updater patch


def patch_updater(binary: Path) -> int:
    """Point the Swift updater's release feed at an unresolvable host."""
    assert len(UPDATE_URL_OLD) == len(UPDATE_URL_NEW)
    blob = binary.read_bytes()
    count = blob.count(UPDATE_URL_OLD)
    if count == 0:
        raise DonorMismatch(
            f"release URL not found in {binary.name}; the updater changed, "
            "review the update-disable mechanism before building"
        )
    binary.write_bytes(blob.replace(UPDATE_URL_OLD, UPDATE_URL_NEW))
    return count


# ------------------------------------------------------------------- signing


def _is_macho(path: Path) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        with open(path, "rb") as fh:
            return fh.read(4) in MACHO_MAGICS
    except OSError:
        return False


def find_macho_files(app: Path) -> list[Path]:
    found = []
    for root, _dirs, files in os.walk(app):
        for fname in files:
            p = Path(root) / fname
            if _is_macho(p):
                found.append(p)
    return sorted(found)


def _entitlements_xml(path: Path, runner: Runner) -> bytes:
    proc = runner(["codesign", "-d", "--entitlements", ":-", str(path)])
    out = proc.stdout or b""
    return out if out.lstrip().startswith(b"<?xml") else b""


def sign_bundle(app: Path, *, runner: Runner = default_runner, workers: int = 8) -> None:
    """Ad-hoc re-sign inside-out: libraries, executables, scripts, then bundle.

    Executables and the bundle keep the donor's hardened runtime and
    entitlements (allow-jit, allow-unsigned-executable-memory and, crucially,
    disable-library-validation, which lets the hardened runtime load the
    ad-hoc signed dylibs). Plain libraries get no runtime flag. No ``--deep``.
    """
    macho = find_macho_files(app)
    libs = [p for p in macho if p.name.endswith(LIB_SUFFIXES)]
    execs = [p for p in macho if not p.name.endswith(LIB_SUFFIXES)]
    main_exec = app / "Contents" / "MacOS" / "oMLX"

    with tempfile.TemporaryDirectory(prefix="fork-ent-") as tmp:
        tmpdir = Path(tmp)
        # Read entitlements first: signing replaces the donor signature.
        ent_files: dict[Path, Path] = {}
        for i, exe in enumerate(execs + ([main_exec] if main_exec not in execs else [])):
            xml = _entitlements_xml(exe, runner)
            if xml:
                f = tmpdir / f"ent-{i}.plist"
                f.write_bytes(xml)
                ent_files[exe] = f

        def sign(target: Path, *, runtime: bool, ent: Path | None):
            cmd = ["codesign", "--force", "--timestamp=none", "-s", "-"]
            if runtime:
                cmd += ["--options", "runtime"]
            if ent is not None:
                cmd += ["--entitlements", str(ent)]
            cmd.append(str(target))
            _check(runner(cmd), f"codesign {target.name}")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda p: sign(p, runtime=False, ent=None), libs))
        for exe in execs:
            sign(exe, runtime=True, ent=ent_files.get(exe))
        macos = app / "Contents" / "MacOS"
        if macos.is_dir():
            for script in sorted(macos.iterdir()):
                if script.is_file() and script not in execs:
                    sign(script, runtime=False, ent=None)
        sign(app, runtime=True, ent=ent_files.get(main_exec))


def verify_bundle(app: Path, *, runner: Runner = default_runner) -> dict:
    proc = runner(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)])
    text = ((proc.stdout or b"") + (proc.stderr or b"")).decode(errors="replace").strip()
    return {"ok": proc.returncode == 0, "output": text}


def assess_gatekeeper(app: Path, *, runner: Runner = default_runner) -> str:
    proc = runner(["spctl", "-a", "-vv", str(app)])
    return ((proc.stdout or b"") + (proc.stderr or b"")).decode(errors="replace").strip()


# ----------------------------------------------------------------------- dmg


def make_dmg(app: Path, dmg: Path, *, volname: str, runner: Runner = default_runner) -> None:
    """Drag-to-Applications disk image: the app plus an /Applications symlink."""
    stage = dmg.parent / "_dmg_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        staged_app = stage / app.name
        # APFS clone (instant, no extra disk); ditto as the fallback.
        if runner(["cp", "-cRp", str(app), str(staged_app)]).returncode != 0:
            if staged_app.exists():
                shutil.rmtree(staged_app)
            _check(runner(["ditto", str(app), str(staged_app)]), "ditto (dmg stage)")
        (stage / "Applications").symlink_to("/Applications")
        if dmg.exists():
            dmg.unlink()
        _check(
            runner([
                "hdiutil", "create", "-volname", volname, "-srcfolder", str(stage),
                "-ov", "-format", "UDZO", str(dmg),
            ]),
            "hdiutil create",
        )
        _check(runner(["hdiutil", "verify", str(dmg)]), "hdiutil verify")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


# --------------------------------------------------------------------- build


def _git_rev(repo: Path, runner: Runner) -> str | None:
    try:
        proc = runner(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"])
    except OSError:
        return None
    return proc.stdout.decode().strip() if proc.returncode == 0 else None


def _dir_size(path: Path) -> int:
    total = 0
    for root, _d, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def build(
    opts: Options,
    *,
    venv_pins: Mapping[str, str] | None = None,
    runner: Runner = default_runner,
) -> dict:
    donor = opts.donor
    pins = dict(venv_pins) if venv_pins is not None else read_venv_pins()
    validate_donor(donor, opts.expect_version, opts.expect_build, pins)

    out = opts.out
    out.mkdir(parents=True, exist_ok=True)
    app = out / f"{opts.name}.app"
    if app.exists():
        shutil.rmtree(app)

    _check(runner(["ditto", str(donor), str(app)]), "ditto (copy donor)")

    contents = app / "Contents"
    overlay = overlay_package(opts.package, contents / "Resources" / "omlx")

    info = patch_info_plist(
        contents / "Info.plist",
        name=opts.name,
        bundle_id=opts.bundle_id,
        fork_rev=opts.fork_rev,
        env=opts.env,
    )
    patched_urls = patch_updater(contents / "MacOS" / "oMLX")

    donor_info = read_info(donor)
    (contents / "Resources" / "FORK_BUILD.json").write_text(
        json.dumps(
            {
                "donor_version": donor_info["CFBundleShortVersionString"],
                "donor_build": donor_info["CFBundleVersion"],
                "fork_version": info["CFBundleShortVersionString"],
                "fork_commit": _git_rev(opts.package.parent, runner),
                "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "env": dict(opts.env),
            },
            indent=2,
        )
        + "\n"
    )

    runner(["xattr", "-dr", "com.apple.quarantine", str(app)])

    verify = None
    gatekeeper = None
    if opts.sign:
        sign_bundle(app, runner=runner)
        verify = verify_bundle(app, runner=runner)
        gatekeeper = assess_gatekeeper(app, runner=runner)

    dmg = None
    if opts.dmg:
        dmg = out / f"{opts.name.replace(' ', '-')}-{info['CFBundleShortVersionString']}.dmg"
        make_dmg(app, dmg, volname=opts.name, runner=runner)

    return {
        "app": app,
        "dmg": dmg,
        "app_size": _dir_size(app),
        "dmg_size": dmg.stat().st_size if dmg and dmg.exists() else None,
        "bundle_id": info["CFBundleIdentifier"],
        "version": info["CFBundleShortVersionString"],
        "overlay": overlay,
        "updater_urls_patched": patched_urls,
        "verify": verify,
        "gatekeeper": gatekeeper,
    }


# ----------------------------------------------------------------------- CLI


def parse_args(argv=None) -> Options:
    d = Options()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--donor", type=Path, default=d.donor)
    p.add_argument("--out", type=Path, default=d.out)
    p.add_argument("--name", default=d.name)
    p.add_argument("--bundle-id", default=d.bundle_id)
    p.add_argument("--package", type=Path, default=d.package,
                   help="fork Python package to embed (default: repo omlx/)")
    p.add_argument("--expect-version", default=d.expect_version,
                   help="donor CFBundleShortVersionString to accept")
    p.add_argument("--expect-build", default=d.expect_build,
                   help="donor CFBundleVersion to accept")
    p.add_argument("--fork-rev", type=int, default=d.fork_rev,
                   help="N in the 0.7.0-fork.N version suffix")
    p.add_argument("--no-dmg", dest="dmg", action="store_false")
    p.add_argument("--no-sign", dest="sign", action="store_false")
    ns = p.parse_args(argv)
    return Options(
        donor=ns.donor, out=ns.out, name=ns.name, bundle_id=ns.bundle_id,
        package=ns.package, expect_version=ns.expect_version,
        expect_build=ns.expect_build, fork_rev=ns.fork_rev,
        sign=ns.sign, dmg=ns.dmg,
    )


def _human(n: int | None) -> str:
    if n is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return str(n)


def print_summary(s: dict) -> None:
    ov = s["overlay"]
    print("== oMLX Fork build summary ==")
    print(f"app:        {s['app']}  ({_human(s['app_size'])})")
    print(f"dmg:        {s['dmg']}  ({_human(s['dmg_size'])})" if s["dmg"] else "dmg:        (skipped)")
    print(f"bundle id:  {s['bundle_id']}")
    print(f"version:    {s['version']}")
    print(f"overlay:    {ov['changed']} changed, {ov['added']} added, "
          f"{ov['pyc_removed']} stale .pyc removed, "
          f"{len(ov['donor_only_py'])} donor-only .py kept")
    print(f"updater:    {s['updater_urls_patched']} release URL(s) redirected")
    if s["verify"] is None:
        print("signature:  (skipped, --no-sign)")
    else:
        print(f"signature:  {'OK' if s['verify']['ok'] else 'FAILED'}")
        print(s["verify"]["output"])
        print("spctl (informational; ad-hoc is rejected by Gatekeeper assessment):")
        print(s["gatekeeper"])


def main(argv=None) -> int:
    opts = parse_args(argv)
    try:
        summary = build(opts)
    except (DonorMismatch, BuildError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print_summary(summary)
    if summary["verify"] is not None and not summary["verify"]["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

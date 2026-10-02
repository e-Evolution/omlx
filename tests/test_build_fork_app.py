# SPDX-License-Identifier: Apache-2.0
"""Tests for packaging/build_fork_app.py against a synthetic donor bundle.

The module is loaded by file path: the repo's ``packaging/`` directory would
otherwise shadow the PyPI ``packaging`` distribution that omlx imports.
No test runs the real codesign/hdiutil; a recording runner stands in.
"""

import importlib.util
import json
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parent.parent / "packaging" / "build_fork_app.py"
URL = b"https://api.github.com/repos/jundot/omlx/releases?per_page=20"
MACHO = bytes.fromhex("cffaedfe") + b"\x00" * 60

PINS = {
    "mlx": "0.32.2",
    "mlx-metal": "0.32.2",
    "mlx-lm": "0.31.4.dev132+g94cdcae13",
    "mlx-vlm": "0.7.1",
    "dflash-mlx": "0.1.10+omlx.9",
    "transformers": "5.17.0",
    "fastapi": "0.142.2",
}


def _load():
    spec = importlib.util.spec_from_file_location("build_fork_app", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_fork_app"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bfa():
    return _load()


class FakeRunner:
    """Records codesign/hdiutil/spctl; delegates ditto/xattr/cp to the OS."""

    FAKED = ("codesign", "hdiutil", "spctl")

    def __init__(self, entitlements_xml=b""):
        self.calls = []
        self.entitlements_xml = entitlements_xml

    def __call__(self, cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        if cmd[0] in self.FAKED:
            stdout = b""
            if cmd[0] == "codesign" and "-d" in cmd and any(
                a.startswith("--entitlements") for a in cmd
            ):
                stdout = self.entitlements_xml
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=b"")
        return subprocess.run(cmd, capture_output=True, **{
            k: v for k, v in kwargs.items() if k != "capture_output"
        })

    def codesign_targets(self):
        return [
            c[-1] for c in self.calls
            if c[0] == "codesign" and "--force" in c
        ]


def _write(path: Path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())


@pytest.fixture
def donor(tmp_path):
    app = tmp_path / "donor" / "oMLX.app"
    contents = app / "Contents"
    _write(
        contents / "Info.plist",
        plistlib.dumps(
            {
                "CFBundleIdentifier": "app.omlx",
                "CFBundleName": "oMLX",
                "CFBundleDisplayName": "oMLX",
                "CFBundleExecutable": "oMLX",
                "CFBundleShortVersionString": "0.7.0",
                "CFBundleVersion": "2987",
                "CFBundleURLTypes": [
                    {"CFBundleURLName": "app.omlx.url", "CFBundleURLSchemes": ["omlxapp"]}
                ],
            }
        ),
    )
    _write(contents / "MacOS" / "oMLX", MACHO + b"pad" + URL + b"pad" + URL + b"end")
    _write(contents / "MacOS" / "omlx-cli", "#!/bin/sh\nexec true\n")
    _write(contents / "MacOS" / "omlx-cluster-python", "#!/bin/sh\nexec true\n")
    pkg = contents / "Resources" / "omlx"
    _write(pkg / "__init__.py", "old = 1\n")
    _write(pkg / "scheduler.py", "OLD = 1\n")
    _write(pkg / "__pycache__" / "scheduler.cpython-311.pyc", b"stale")
    _write(pkg / "__pycache__" / "__init__.cpython-311.pyc", b"fresh-keep")
    _write(pkg / "_build_info.py", 'build_number = "donor"\n')
    _write(pkg / "_engine_commits.json", "{}")
    _write(pkg / "custom_kernels" / "bonsai" / "_ext.cpython-311-darwin.so", MACHO + b"donor-so")
    _write(pkg / "custom_kernels" / "bonsai" / "k.metallib", b"donor-metallib")
    sp = (
        contents / "Resources" / "Python" / "framework-mlx-base" / "lib"
        / "python3.11" / "site-packages"
    )
    for name, version in PINS.items():
        _write(sp / f"{name.replace('-', '_')}-{version}.dist-info" / "METADATA", "x")
    _write(contents / "Resources" / "Python" / "cpython-3.11" / "bin" / "python3.11", MACHO)
    _write(contents / "Resources" / "Python" / "cpython-3.11" / "lib" / "libpython3.11.dylib", MACHO)
    _write(sp / "mlx" / "core.cpython-311-darwin.so", MACHO)
    _write(contents / "Resources" / "AppIcon.icns", "icon")
    return app


@pytest.fixture
def fork_pkg(tmp_path):
    pkg = tmp_path / "repo" / "omlx"
    _write(pkg / "__init__.py", "new = 2\n")
    _write(pkg / "scheduler.py", "NEW = 2\n")
    _write(pkg / "added.py", "ADDED = 1\n")
    _write(pkg / "__pycache__" / "added.cpython-311.pyc", b"junk")
    _write(pkg / "stray.pyc", b"junk")
    _write(pkg / "tests" / "test_x.py", "assert False\n")
    _write(pkg / "sub" / "tests" / "test_y.py", "assert False\n")
    _write(pkg / ".git" / "HEAD", "ref")
    _write(pkg / "custom_kernels" / "bonsai" / "csrc" / "k.cpp", "src")
    _write(pkg / "custom_kernels" / "bonsai" / "_ext.cpython-311-darwin.so", b"REPO-so")
    _write(pkg / "custom_kernels" / "bonsai" / "__init__.py", "")
    return pkg


# ---------------------------------------------------------------- validation


def test_read_donor_pins_parses_dist_info(bfa, donor):
    pins = bfa.read_donor_pins(donor)
    assert pins == PINS


def test_validate_donor_passes_when_everything_matches(bfa, donor):
    bfa.validate_donor(donor, "0.7.0", "2987", dict(PINS))


def test_validate_donor_rejects_wrong_version(bfa, donor):
    with pytest.raises(bfa.DonorMismatch, match="0.7.0"):
        bfa.validate_donor(donor, "0.8.0", "2987", dict(PINS))


def test_validate_donor_rejects_wrong_build(bfa, donor):
    with pytest.raises(bfa.DonorMismatch, match="2987"):
        bfa.validate_donor(donor, "0.7.0", "9999", dict(PINS))


def test_validate_donor_rejects_pin_mismatch_and_names_package(bfa, donor):
    venv = dict(PINS, mlx="0.33.0")
    with pytest.raises(bfa.DonorMismatch, match="mlx"):
        bfa.validate_donor(donor, "0.7.0", "2987", venv)


def test_validate_donor_rejects_missing_donor_pin(bfa, donor):
    sp = next(donor.rglob("site-packages"))
    for d in sp.glob("fastapi-*.dist-info"):
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    with pytest.raises(bfa.DonorMismatch, match="fastapi"):
        bfa.validate_donor(donor, "0.7.0", "2987", dict(PINS))


def test_validate_donor_rejects_sparkle(bfa, donor):
    plist = donor / "Contents" / "Info.plist"
    data = plistlib.loads(plist.read_bytes())
    data["SUFeedURL"] = "https://x"
    plist.write_bytes(plistlib.dumps(data))
    with pytest.raises(bfa.DonorMismatch, match="SU"):
        bfa.validate_donor(donor, "0.7.0", "2987", dict(PINS))


# ------------------------------------------------------------------ overlay


def test_overlay_replaces_changed_and_adds_new_files(bfa, donor, fork_pkg):
    dest = donor / "Contents" / "Resources" / "omlx"
    bfa.overlay_package(fork_pkg, dest)
    assert (dest / "scheduler.py").read_text() == "NEW = 2\n"
    assert (dest / "__init__.py").read_text() == "new = 2\n"
    assert (dest / "added.py").read_text() == "ADDED = 1\n"


def test_overlay_applies_the_donor_file_filter(bfa, donor, fork_pkg):
    dest = donor / "Contents" / "Resources" / "omlx"
    bfa.overlay_package(fork_pkg, dest)
    assert not (dest / "tests").exists()
    assert not (dest / "sub" / "tests").exists()
    assert not (dest / ".git").exists()
    assert not (dest / "stray.pyc").exists()
    assert not (dest / "__pycache__" / "added.cpython-311.pyc").exists()
    assert not (dest / "custom_kernels" / "bonsai" / "csrc").exists()


def test_overlay_keeps_donor_only_build_artifacts(bfa, donor, fork_pkg):
    dest = donor / "Contents" / "Resources" / "omlx"
    bfa.overlay_package(fork_pkg, dest)
    assert (dest / "_build_info.py").read_text() == 'build_number = "donor"\n'
    assert (dest / "_engine_commits.json").exists()
    so = dest / "custom_kernels" / "bonsai" / "_ext.cpython-311-darwin.so"
    assert so.read_bytes().endswith(b"donor-so")  # repo's .so never wins
    assert (dest / "custom_kernels" / "bonsai" / "k.metallib").exists()


def test_overlay_drops_only_stale_pyc_of_changed_sources(bfa, donor, fork_pkg):
    dest = donor / "Contents" / "Resources" / "omlx"
    # __init__.py differs too, so both change; add an unchanged module.
    _write(dest / "same.py", "S = 1\n")
    _write(fork_pkg / "same.py", "S = 1\n")
    _write(dest / "__pycache__" / "same.cpython-311.pyc", b"keep-me")
    stats = bfa.overlay_package(fork_pkg, dest)
    assert not (dest / "__pycache__" / "scheduler.cpython-311.pyc").exists()
    assert (dest / "__pycache__" / "same.cpython-311.pyc").exists()
    assert stats["pyc_removed"] >= 1
    assert stats["changed"] >= 2


def test_overlay_is_idempotent(bfa, donor, fork_pkg):
    dest = donor / "Contents" / "Resources" / "omlx"
    bfa.overlay_package(fork_pkg, dest)
    stats = bfa.overlay_package(fork_pkg, dest)
    assert stats["changed"] == 0 and stats["added"] == 0


# -------------------------------------------------------------------- plist


def test_patch_info_plist_sets_identity_version_and_env(bfa, donor):
    plist = donor / "Contents" / "Info.plist"
    bfa.patch_info_plist(
        plist, name="oMLX Fork", bundle_id="app.omlx.fork", fork_rev=1,
        env={"OMLX_MEMORY_PRESSURE_PARK": "true"},
    )
    data = plistlib.loads(plist.read_bytes())
    assert data["CFBundleIdentifier"] == "app.omlx.fork"
    assert data["CFBundleName"] == "oMLX Fork"
    assert data["CFBundleDisplayName"] == "oMLX Fork"
    assert data["CFBundleShortVersionString"] == "0.7.0-fork.1"
    assert data["CFBundleVersion"] == "2987"
    assert data["CFBundleExecutable"] == "oMLX"  # binary name never changes
    assert data["LSEnvironment"] == {"OMLX_MEMORY_PRESSURE_PARK": "true"}
    assert data["CFBundleURLTypes"][0]["CFBundleURLSchemes"] == ["omlxapp"]


def test_patch_info_plist_is_idempotent_on_the_suffix(bfa, donor):
    plist = donor / "Contents" / "Info.plist"
    for _ in range(2):
        bfa.patch_info_plist(plist, name="n", bundle_id="b.c", fork_rev=2, env={})
    data = plistlib.loads(plist.read_bytes())
    assert data["CFBundleShortVersionString"] == "0.7.0-fork.2"


def test_fork_version_keeps_the_numeric_prefix_the_swift_parser_reads(bfa):
    # ReleasesChecker.parseVersion splits on the first non [0-9.] character and
    # looks for dev/alpha/beta/rc markers (a/b only next to digits or . -).
    version = bfa.fork_version("0.7.0", 1)
    head = version.replace("-", " ").split()[0]
    assert head == "0.7.0"
    low = version.lower()
    for marker in ("dev", "alpha", "beta", "rc"):
        assert marker not in low
    assert "a" not in low and "b" not in low


def test_default_env_enables_d1_and_park(bfa):
    assert bfa.FORK_ENV == {
        "OMLX_PREFILL_PROGRESS_COMMIT_BLOCKS": "4",
        "OMLX_MEMORY_PRESSURE_PARK": "true",
        "OMLX_MLX_CACHE_LIMIT_GB": "16",
        "OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS": "200000",
    }
    assert bfa.FORK_MLX_CACHE_LIMIT_GB == "16"
    assert bfa.FORK_MAX_BATCHED_DECODE_CONTEXT_TOKENS == "200000"


# ------------------------------------------------------------ updater patch


def test_patch_updater_redirects_every_release_url_same_length(bfa, donor):
    binary = donor / "Contents" / "MacOS" / "oMLX"
    before = binary.read_bytes()
    count = bfa.patch_updater(binary)
    after = binary.read_bytes()
    assert count == 2
    assert len(after) == len(before)
    assert URL not in after
    assert b"api.github.com" not in after
    assert b"https://no-update.test/repos/jundot/omlx/releases?per_page=20" in after


def test_patch_updater_fails_loudly_when_url_not_found(bfa, tmp_path):
    binary = tmp_path / "oMLX"
    binary.write_bytes(MACHO + b"nothing here")
    with pytest.raises(bfa.DonorMismatch, match="release URL"):
        bfa.patch_updater(binary)


# ------------------------------------------------------------------ signing


def test_find_macho_files_detects_by_magic(bfa, donor):
    found = {p.name for p in bfa.find_macho_files(donor)}
    assert {"oMLX", "python3.11", "libpython3.11.dylib",
            "core.cpython-311-darwin.so", "_ext.cpython-311-darwin.so"} <= found
    assert "omlx-cli" not in found and "AppIcon.icns" not in found


ENT = (b'<?xml version="1.0"?><plist version="1.0"><dict><key>'
       b'com.apple.security.cs.allow-jit</key><true/></dict></plist>')


def test_sign_bundle_goes_inside_out_with_hardened_runtime(bfa, donor):
    runner = FakeRunner(entitlements_xml=ENT)
    bfa.sign_bundle(donor, runner=runner)
    order = runner.codesign_targets()
    app = str(donor)
    assert order[-1] == app
    libs = [i for i, t in enumerate(order) if t.endswith((".so", ".dylib"))]
    execs = [i for i, t in enumerate(order)
             if t.endswith(("/MacOS/oMLX", "python3.11"))]
    assert libs and execs and max(libs) < min(execs)
    assert order.index(app) > max(execs)


def test_sign_bundle_uses_adhoc_identity_runtime_and_donor_entitlements(bfa, donor):
    runner = FakeRunner(entitlements_xml=ENT)
    bfa.sign_bundle(donor, runner=runner)
    signs = [c for c in runner.calls if c[0] == "codesign" and "--force" in c]
    assert all(c[c.index("-s") + 1] == "-" for c in signs)
    bundle = next(c for c in signs if c[-1] == str(donor))
    assert "--options" in bundle and bundle[bundle.index("--options") + 1] == "runtime"
    assert "--entitlements" in bundle
    ent_path = Path(bundle[bundle.index("--entitlements") + 1])
    assert ent_path.name  # path handed to codesign
    assert "--deep" not in bundle
    # shared libraries never carry the runtime flag
    lib = next(c for c in signs if c[-1].endswith(".dylib"))
    assert "--options" not in lib


def test_verify_bundle_runs_deep_strict(bfa, donor):
    runner = FakeRunner()
    result = bfa.verify_bundle(donor, runner=runner)
    cmd = next(c for c in runner.calls if "--verify" in c)
    assert "--deep" in cmd and "--strict" in cmd and cmd[-1] == str(donor)
    assert result["ok"] is True


# ---------------------------------------------------------------------- dmg


def test_make_dmg_stages_app_and_applications_symlink(bfa, donor, tmp_path):
    runner = FakeRunner()
    dmg = tmp_path / "out" / "oMLX-Fork-0.7.0-fork.1.dmg"
    dmg.parent.mkdir()
    staged = {}

    def spying(cmd, **kw):
        if cmd[0] == "hdiutil" and cmd[1] == "create":
            stage = Path(cmd[cmd.index("-srcfolder") + 1])
            staged["entries"] = sorted(p.name for p in stage.iterdir())
            staged["link"] = (stage / "Applications").readlink()
        return runner(cmd, **kw)

    bfa.make_dmg(donor, dmg, volname="oMLX Fork", runner=spying)
    assert staged["entries"] == ["Applications", "oMLX.app"]
    assert str(staged["link"]) == "/Applications"
    create = next(c for c in runner.calls if c[:2] == ["hdiutil", "create"])
    assert create[-1] == str(dmg)
    assert "UDZO" in create


# --------------------------------------------------------------- end-to-end


def test_build_end_to_end_without_sign_or_dmg(bfa, donor, fork_pkg, tmp_path):
    out = tmp_path / "dist"
    runner = FakeRunner()
    summary = bfa.build(
        bfa.Options(
            donor=donor, out=out, name="oMLX Fork", bundle_id="app.omlx.fork",
            package=fork_pkg, expect_version="0.7.0", expect_build="2987",
            fork_rev=1, sign=False, dmg=False,
        ),
        venv_pins=dict(PINS), runner=runner,
    )
    app = out / "oMLX Fork.app"
    assert summary["app"] == app and summary["dmg"] is None
    data = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert data["CFBundleIdentifier"] == "app.omlx.fork"
    assert data["LSEnvironment"]["OMLX_MEMORY_PRESSURE_PARK"] == "true"
    assert (app / "Contents" / "Resources" / "omlx" / "added.py").exists()
    assert b"api.github.com" not in (app / "Contents" / "MacOS" / "oMLX").read_bytes()
    marker = json.loads((app / "Contents" / "Resources" / "FORK_BUILD.json").read_text())
    assert marker["donor_version"] == "0.7.0" and marker["fork_version"] == "0.7.0-fork.1"
    assert not [c for c in runner.calls if c[0] in ("codesign", "hdiutil")]


def test_build_never_modifies_the_donor(bfa, donor, fork_pkg, tmp_path):
    def snapshot(root):
        return {
            str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()
        }

    before = snapshot(donor)
    bfa.build(
        bfa.Options(donor=donor, out=tmp_path / "dist", package=fork_pkg,
                    sign=False, dmg=False),
        venv_pins=dict(PINS), runner=FakeRunner(),
    )
    assert snapshot(donor) == before


def test_build_aborts_before_copying_when_pins_mismatch(bfa, donor, fork_pkg, tmp_path):
    out = tmp_path / "dist"
    with pytest.raises(bfa.DonorMismatch):
        bfa.build(
            bfa.Options(donor=donor, out=out, package=fork_pkg, sign=False, dmg=False),
            venv_pins=dict(PINS, transformers="9.9.9"), runner=FakeRunner(),
        )
    assert not (out / "oMLX Fork.app").exists()


def test_build_replaces_a_previous_output_bundle(bfa, donor, fork_pkg, tmp_path):
    out = tmp_path / "dist"
    stale = out / "oMLX Fork.app" / "Contents" / "stale.txt"
    _write(stale, "old")
    _write(out / "oMLX Fork.app" / bfa.BUILD_MARKER, "m")
    bfa.build(
        bfa.Options(donor=donor, out=out, package=fork_pkg, sign=False, dmg=False),
        venv_pins=dict(PINS), runner=FakeRunner(),
    )
    assert not stale.exists()


def test_cli_parses_documented_flags(bfa):
    opts = bfa.parse_args([
        "--donor", "/Applications/oMLX.app", "--out", "x", "--name", "N",
        "--bundle-id", "a.b", "--no-dmg", "--no-sign",
        "--expect-version", "1.0", "--expect-build", "5", "--fork-rev", "3",
    ])
    assert opts.dmg is False and opts.sign is False
    assert opts.name == "N" and opts.bundle_id == "a.b"
    assert opts.expect_version == "1.0" and opts.expect_build == "5"
    assert opts.fork_rev == 3
    defaults = bfa.parse_args([])
    assert defaults.donor == Path("/Applications/oMLX.app")
    assert defaults.bundle_id == "app.omlx.fork" and defaults.name == "oMLX Fork"
    assert defaults.expect_version == "0.7.0" and defaults.expect_build == "2987"
    assert defaults.sign is True and defaults.dmg is True


# ------------------------------------------- output / donor safety (review R3)


def _opts(bfa, donor, fork_pkg, out, **kw):
    kw.setdefault("sign", False)
    kw.setdefault("dmg", False)
    return bfa.Options(donor=donor, out=out, package=fork_pkg, **kw)


def test_build_refuses_an_output_bundle_equal_to_the_donor(bfa, donor, fork_pkg):
    before = (donor / "Contents" / "Info.plist").read_bytes()
    with pytest.raises(bfa.BuildError, match="donor"):
        bfa.build(
            _opts(bfa, donor, fork_pkg, donor.parent, name="oMLX"),
            venv_pins=dict(PINS), runner=FakeRunner(),
        )
    assert (donor / "Contents" / "Info.plist").read_bytes() == before


def test_build_refuses_an_output_bundle_inside_the_donor(bfa, donor, fork_pkg):
    with pytest.raises(bfa.BuildError, match="donor"):
        bfa.build(
            _opts(bfa, donor, fork_pkg, donor / "Contents" / "Resources"),
            venv_pins=dict(PINS), runner=FakeRunner(),
        )
    assert not (donor / "Contents" / "Resources" / "oMLX Fork.app").exists()


def test_build_refuses_an_output_bundle_that_contains_the_donor(bfa, donor, fork_pkg):
    # out/<name>.app is an ancestor of the donor: rmtree would delete it.
    out = donor.parent.parent
    wrapper = out / "Wrapper.app"
    wrapper.mkdir()
    nested = wrapper / "inner" / "oMLX.app"
    nested.parent.mkdir()
    donor.rename(nested)
    with pytest.raises(bfa.BuildError, match="donor"):
        bfa.build(
            _opts(bfa, nested, fork_pkg, out, name="Wrapper"),
            venv_pins=dict(PINS), runner=FakeRunner(),
        )
    assert (nested / "Contents" / "Info.plist").is_file()


def test_build_refuses_the_applications_directories(bfa, donor, fork_pkg, monkeypatch, tmp_path):
    # Never point this at the real /Applications: with a regression it would
    # write there. The script resolves the protected dirs through one helper.
    fake_system = tmp_path / "Applications"
    fake_user = tmp_path / "home" / "Applications"
    fake_system.mkdir()
    fake_user.mkdir(parents=True)
    monkeypatch.setattr(bfa, "protected_output_dirs", lambda: (fake_system, fake_user))
    for out in (fake_system, fake_user):
        with pytest.raises(bfa.BuildError, match="Applications"):
            bfa.build(
                _opts(bfa, donor, fork_pkg, out),
                venv_pins=dict(PINS), runner=FakeRunner(),
            )
        assert not list(out.iterdir())


def test_protected_output_dirs_are_the_system_and_user_applications(bfa):
    assert Path("/Applications") in bfa.protected_output_dirs()
    assert Path.home() / "Applications" in bfa.protected_output_dirs()


def test_build_does_not_delete_an_existing_bundle_without_the_build_marker(
    bfa, donor, fork_pkg, tmp_path
):
    out = tmp_path / "dist"
    precious = out / "oMLX Fork.app" / "Contents" / "precious.txt"
    _write(precious, "mine")
    with pytest.raises(bfa.BuildError, match="marker"):
        bfa.build(
            _opts(bfa, donor, fork_pkg, out),
            venv_pins=dict(PINS), runner=FakeRunner(),
        )
    assert precious.read_text() == "mine"


def test_build_writes_the_marker_that_allows_the_next_replacement(
    bfa, donor, fork_pkg, tmp_path
):
    out = tmp_path / "dist"
    bfa.build(_opts(bfa, donor, fork_pkg, out), venv_pins=dict(PINS), runner=FakeRunner())
    assert (out / "oMLX Fork.app" / "Contents" / "Resources" / ".omlx-fork-build").is_file()


class FailingVerifyRunner(FakeRunner):
    def __call__(self, cmd, **kwargs):
        if cmd[0] == "codesign" and "--verify" in [str(c) for c in cmd]:
            self.calls.append([str(c) for c in cmd])
            return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"bad seal")
        return super().__call__(cmd, **kwargs)


def test_failed_codesign_verify_aborts_before_any_dmg(bfa, donor, fork_pkg, tmp_path):
    out = tmp_path / "dist"
    runner = FailingVerifyRunner()
    with pytest.raises(bfa.BuildError, match="bad seal"):
        bfa.build(
            _opts(bfa, donor, fork_pkg, out, sign=True, dmg=True),
            venv_pins=dict(PINS), runner=runner,
        )
    assert not [c for c in runner.calls if c[0] == "hdiutil"]
    assert not list(out.glob("*.dmg"))


def test_main_exits_non_zero_when_verify_fails(bfa, monkeypatch, capsys):
    def boom(opts):
        raise bfa.BuildError("codesign verify failed (exit 1): bad seal")

    monkeypatch.setattr(bfa, "build", boom)
    assert bfa.main([]) != 0
    assert "bad seal" in capsys.readouterr().err


def test_read_venv_pins_names_a_missing_package(bfa, monkeypatch):
    import importlib.metadata as md

    def fake_version(name):
        if name == "mlx-vlm":
            raise md.PackageNotFoundError(name)
        return "1"

    monkeypatch.setattr(bfa.importlib.metadata, "version", fake_version)
    with pytest.raises(bfa.BuildError, match="mlx-vlm"):
        bfa.read_venv_pins()


def test_overlay_does_not_follow_symlinked_dirs_out_of_the_bundle(bfa, donor, fork_pkg, tmp_path):
    outside = tmp_path / "outside"
    _write(outside / "leak.py", "LEAK = 1\n")
    (fork_pkg / "linkdir").symlink_to(outside, target_is_directory=True)
    dest = donor / "Contents" / "Resources" / "omlx"
    bfa.overlay_package(fork_pkg, dest)
    assert not (dest / "linkdir" / "leak.py").exists()
    assert not (outside / "extra").exists()


def test_overlay_removes_stale_py_modules_but_keeps_donor_build_products(
    bfa, donor, fork_pkg
):
    dest = donor / "Contents" / "Resources" / "omlx"
    _write(dest / "deleted_in_repo.py", "STALE = 1\n")
    _write(dest / "pkg" / "gone.py", "STALE = 1\n")
    stats = bfa.overlay_package(fork_pkg, dest)
    assert not (dest / "deleted_in_repo.py").exists()
    assert not (dest / "pkg" / "gone.py").exists()
    assert (dest / "_build_info.py").is_file()
    assert (dest / "_engine_commits.json").is_file()
    assert (dest / "custom_kernels" / "bonsai" / "_ext.cpython-311-darwin.so").is_file()
    assert "deleted_in_repo.py" in stats["stale_py_removed"]

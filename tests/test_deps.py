"""MediaPipe availability: the platform pins, and life without them.

Nothing here imports mediapipe. That is the point -- these are the two
failure modes that only show up on a board you cannot get at from CI
(the wrong interpreter, or no wheel at all), so they are tested by
resolving the pins on paper and by making the detector unbuildable.

The wheel facts the pins encode, from PyPI, as of mediapipe 1.0.1:

  0.10.18   last aarch64 linux release before the gap; cp39-cp312 only
  0.10.20+  no aarch64 linux wheels at all
  0.10.30+  py3-none, but macOS arm64 / linux x86_64 / windows only
  1.0.0+    py3-none, and aarch64 linux is back

py3-none means no ABI tag, so 1.0 installs on any Python 3 including
3.13, while 0.10.18's cp312 wheel does not.
"""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from tlod.config import Config
from tlod.vision.hands import HandDetector, NullHandDetector

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _hands_requirements() -> list[Requirement]:
    data = tomllib.loads(PYPROJECT.read_text())
    return [Requirement(r) for r in data["project"]["optional-dependencies"]["hands"]]


def _environment(sys_platform: str, machine: str, python_version: str) -> dict[str, str]:
    """The PEP 508 marker variables. All of them: packaging raises on an
    undefined name rather than treating it as false."""
    return {
        "sys_platform": sys_platform,
        "platform_machine": machine,
        "platform_system": {"linux": "Linux", "darwin": "Darwin"}[sys_platform],
        "platform_release": "",
        "os_name": "posix",
        "python_version": python_version,
        "python_full_version": python_version + ".0",
        "implementation_name": "cpython",
        "implementation_version": python_version + ".0",
        "platform_python_implementation": "CPython",
        "extra": "hands",
    }


def _applicable(sys_platform: str, machine: str, python_version: str) -> list[Requirement]:
    env = _environment(sys_platform, machine, python_version)
    return [r for r in _hands_requirements() if r.marker is None or r.marker.evaluate(env)]


@pytest.mark.parametrize(
    "platform,machine,python",
    [("linux", "aarch64", "3.12"), ("linux", "aarch64", "3.13"),
     ("linux", "x86_64", "3.12"), ("linux", "x86_64", "3.13"),
     ("darwin", "arm64", "3.12"), ("darwin", "arm64", "3.13")],
)
def test_every_supported_platform_gets_exactly_one_pin(platform, machine, python):
    """Two matching lines would be an unsatisfiable conflict, zero would
    silently install nothing at all and only fail at `import mediapipe`."""
    assert len(_applicable(platform, machine, python)) == 1


def test_orange_pi_on_python_313_can_install_something():
    """The bug this file exists for: 0.10.x has no cp313 aarch64 wheel, so
    pinning below 1.0 there resolves to "no matching distribution"."""
    spec = _applicable("linux", "aarch64", "3.13")[0].specifier
    assert spec.contains(Version("1.0.1"))
    assert not spec.contains(Version("0.10.18"))


def test_orange_pi_on_python_312_keeps_the_build_known_to_work():
    """3.13 has to move; 3.12 does not, and an untested jump to 1.0 on a
    board that is already running is not an upgrade."""
    spec = _applicable("linux", "aarch64", "3.12")[0].specifier
    assert spec.contains(Version("0.10.18"))
    assert not spec.contains(Version("1.0.1"))


def test_macos_still_excludes_the_version_that_aborts():
    """1.0 on macOS arm64 is not a bug to work around in Python: the palm
    detector CHECK-fails inside Metal and the process dies."""
    spec = _applicable("darwin", "arm64", "3.13")[0].specifier
    assert not spec.contains(Version("1.0.1"))
    assert spec.contains(Version("0.10.35"))


# --------------------------------------------------------------------------
# degrading without mediapipe
# --------------------------------------------------------------------------


def test_null_detector_is_a_hand_detector():
    det = NullHandDetector()
    assert isinstance(det, HandDetector)
    assert det.detect(object()) == []
    det.close()


def test_detector_none_skips_mediapipe_entirely():
    from tlod.cli import build_detector

    cfg = Config().with_overrides(vision={"detector": "none"})
    assert isinstance(build_detector(cfg), NullHandDetector)


@pytest.mark.parametrize("failure", [ImportError("No module named 'mediapipe'"),
                                     RuntimeError("GPU service not available")])
def test_a_missing_or_unusable_mediapipe_costs_the_hands_and_nothing_else(
    monkeypatch, caplog, failure
):
    """Whether the wheel is absent or present-but-unable-to-build-a-graph,
    `vision-serve` on the board should keep publishing objects rather than
    exit on a traceback -- and should say why."""
    import tlod.vision.hands as hands
    from tlod.cli import build_detector

    def explode(**kwargs):
        raise failure

    monkeypatch.setattr(hands, "MediaPipeHandDetector", explode)
    with caplog.at_level("WARNING"):
        det = build_detector(Config())

    assert isinstance(det, NullHandDetector)
    assert "no hand detector available" in caplog.text
    assert type(failure).__name__ in caplog.text

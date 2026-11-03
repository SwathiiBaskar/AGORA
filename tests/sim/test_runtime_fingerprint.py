import json
import os
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from civbox.cli import EXIT_OK, main
from civbox.core.canonical import canonical_json
from civbox.sim.runtime import DISPATCH_ENVIRONMENT_VARIABLE, RuntimeFingerprint


def fingerprint(**changes: Any) -> RuntimeFingerprint:
    fields: dict[str, Any] = {
        "python_implementation": "CPython",
        "python_version": "3.12.3",
        "system": "Linux",
        "machine": "x86_64",
        "byte_order": "little",
        "numpy_version": "2.5.4",
        "cpu_introspection": True,
        "cpu_baseline": ("X86_V2",),
        "cpu_dispatch_available": ("X86_V3", "X86_V4"),
        "cpu_dispatch_active": ("X86_V3", "X86_V4"),
        "npy_disable_cpu_features": None,
    }
    fields.update(changes)
    return RuntimeFingerprint(**fields)


def test_collect_describes_the_current_process() -> None:
    current = RuntimeFingerprint.collect()
    assert current.numpy_version == np.__version__
    assert current.python_version.split(".")[:2] == [
        str(sys.version_info[0]),
        str(sys.version_info[1]),
    ]
    assert current.byte_order == sys.byteorder
    assert current.python_implementation == "CPython"
    assert current.cpu_introspection is True
    assert set(current.cpu_dispatch_active) <= set(current.cpu_dispatch_available)


def test_the_fingerprint_is_plain_json_data_that_round_trips() -> None:
    current = RuntimeFingerprint.collect()
    canonical_json(current.to_dict())
    assert RuntimeFingerprint.from_dict(json.loads(json.dumps(current.to_dict()))) == current


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {**fingerprint().to_dict(), "extra": 1},
        {**fingerprint().to_dict(), "cpu_introspection": "yes"},
        {k: v for k, v in fingerprint().to_dict().items() if k != "machine"},
    ],
)
def test_from_dict_rejects_malformed_data(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RuntimeFingerprint.from_dict(bad)


def test_a_fingerprint_is_immutable() -> None:
    with pytest.raises(ValidationError):
        fingerprint().machine = "arm64"  # type: ignore[misc]


def test_identical_fingerprints_have_no_differences() -> None:
    assert fingerprint().differences(fingerprint()) == ()


def test_differences_name_the_changed_fields_in_declaration_order() -> None:
    other = fingerprint(numpy_version="2.6.0", machine="arm64", cpu_dispatch_active=())
    assert fingerprint().differences(other) == ("machine", "numpy_version", "cpu_dispatch_active")


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"cpu_dispatch_active": ()}, True),
        ({"cpu_dispatch_active": ("X86_V3",)}, False),
        ({}, False),
        # Dispatch state unreadable: nothing can be vouched for, even with no active targets.
        ({"cpu_introspection": False, "cpu_dispatch_active": ()}, False),
    ],
)
def test_restriction_to_baseline_is_only_claimed_when_it_is_confirmed(
    changes: dict[str, Any], expected: bool
) -> None:
    assert fingerprint(**changes).restricted_to_baseline is expected


def test_the_baseline_only_setting_lists_every_available_target() -> None:
    assert fingerprint().baseline_only_setting == "X86_V3 X86_V4"
    assert fingerprint(cpu_dispatch_available=()).baseline_only_setting == ""


# --- the real NumPy, in fresh interpreters ------------------------------------
# NumPy reads NPY_DISABLE_CPU_FEATURES once, when it is first imported, so each case
# needs its own process. The cases adapt to the machine: they compare against what a
# default process reports rather than assuming a particular CPU.


def runtime_report(disable: str | None) -> dict[str, Any]:
    environment = {k: v for k, v in os.environ.items() if k != DISPATCH_ENVIRONMENT_VARIABLE}
    if disable is not None:
        environment[DISPATCH_ENVIRONMENT_VARIABLE] = disable
    result = subprocess.run(
        [sys.executable, "-m", "civbox", "runtime"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report: dict[str, Any] = json.loads(result.stdout)
    return report


def test_a_default_process_records_that_nothing_was_disabled() -> None:
    report = runtime_report(None)
    assert report["fingerprint"]["npy_disable_cpu_features"] is None
    assert report["baseline_only_setting"][DISPATCH_ENVIRONMENT_VARIABLE] == " ".join(
        report["fingerprint"]["cpu_dispatch_available"]
    )


def test_disabling_every_dispatch_target_restricts_numpy_to_its_baseline() -> None:
    default = runtime_report(None)
    setting = default["baseline_only_setting"][DISPATCH_ENVIRONMENT_VARIABLE]
    if not setting:
        pytest.skip("this NumPy build has no dispatch targets to disable")

    pinned = runtime_report(setting)

    assert pinned["restricted_to_baseline"] is True
    assert pinned["fingerprint"]["cpu_dispatch_active"] == []
    assert pinned["fingerprint"]["npy_disable_cpu_features"] == setting


def assert_reports_dispatch_state(
    report: dict[str, Any], default: dict[str, Any], *, setting: str, active: list[str]
) -> None:
    """A report must describe the process it came from.

    An environment variable cannot change which NumPy build is running, so the baseline
    and the available targets match the default process. The requested setting is
    recorded, with an empty value reported as no setting, which is how NumPy treats it.
    The active targets are the expected ones, and the baseline flag agrees with them.
    """
    fingerprint, reference = report["fingerprint"], default["fingerprint"]
    assert fingerprint["cpu_introspection"] is True
    assert fingerprint["cpu_baseline"] == reference["cpu_baseline"]
    assert fingerprint["cpu_dispatch_available"] == reference["cpu_dispatch_available"]
    assert fingerprint["npy_disable_cpu_features"] == (setting or None)
    assert fingerprint["cpu_dispatch_active"] == active
    assert report["restricted_to_baseline"] is (active == [])


def test_restriction_is_reported_target_by_target() -> None:
    default = runtime_report(None)
    available = default["fingerprint"]["cpu_dispatch_available"]
    if not available:
        pytest.skip("this NumPy build has no dispatch targets")

    # Disable every target but the first. With a single target that set is empty, so the
    # setting is an empty value: NumPy must accept it and disable nothing.
    disabled = available[1:]
    setting = " ".join(disabled)
    partial = runtime_report(setting)

    kept = available[0]
    expected = [kept] if kept in default["fingerprint"]["cpu_dispatch_active"] else []
    assert partial["fingerprint"]["cpu_dispatch_active"] == expected
    assert_reports_dispatch_state(partial, default, setting=setting, active=expected)

    if disabled:
        # An empty value is also the boundary of the multi-target case: with no target
        # disabled, every target that is active by default stays active. (With a single
        # target, the case above already is this one.)
        nothing = runtime_report("")
        unchanged = default["fingerprint"]["cpu_dispatch_active"]
        assert_reports_dispatch_state(nothing, default, setting="", active=unchanged)


@pytest.mark.parametrize("mistake", ["lower_case", "unknown_name"])
def test_a_mistyped_setting_does_not_restrict_dispatch_and_the_fingerprint_shows_it(
    mistake: str,
) -> None:
    default = runtime_report(None)
    setting = default["baseline_only_setting"][DISPATCH_ENVIRONMENT_VARIABLE]
    if not setting:
        pytest.skip("this NumPy build has no dispatch targets to disable")
    wrong = setting.lower() if mistake == "lower_case" else "NOT_A_REAL_TARGET"

    report = runtime_report(wrong)

    # NumPy ignores the value with only a hidden ImportWarning, so the process is
    # unrestricted. What was requested is recorded, but what is active is what counts.
    assert report["fingerprint"]["npy_disable_cpu_features"] == wrong
    assert (
        report["fingerprint"]["cpu_dispatch_active"]
        == default["fingerprint"]["cpu_dispatch_active"]
    )
    assert report["restricted_to_baseline"] is default["restricted_to_baseline"]


# --- command -----------------------------------------------------------------


def test_the_runtime_command_prints_the_fingerprint(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["runtime"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["fingerprint"] == RuntimeFingerprint.collect().to_dict()
    assert report["restricted_to_baseline"] is RuntimeFingerprint.collect().restricted_to_baseline


def test_the_runtime_command_does_not_require_valid_settings(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    assert main(["runtime"]) == EXIT_OK
    assert "fingerprint" in json.loads(capsys.readouterr().out)

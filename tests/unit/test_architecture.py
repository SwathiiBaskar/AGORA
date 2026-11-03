"""Module boundaries from ADR-001, enforced on every test run.

Two complementary checks:

* import-linter contracts (``[tool.importlinter]`` in ``pyproject.toml``) forbid named
  dependencies, including indirect ones through other modules.
* An allowlist scan of each constrained package states what it *may* import, so a new,
  unlisted third-party dependency fails here even though no contract names it.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE = REPOSITORY / "src" / "civbox"

# What each package may import: the standard library plus the entries below. Entries are
# dotted prefixes, so "civbox.core" also allows "civbox.core.logging".
ALLOWED: dict[str, frozenset[str]] = {
    "core": frozenset({"civbox.core", "pydantic", "pydantic_settings"}),
    "sim": frozenset({"civbox.core", "civbox.sim", "numpy", "scipy", "pydantic"}),
}


def imported_modules(source: str) -> set[str]:
    """Absolute module names imported anywhere in ``source``, including inside functions."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


def is_allowed(module: str, allowed: frozenset[str]) -> bool:
    root = module.split(".")[0]
    if root in sys.stdlib_module_names:
        return True
    return any(module == entry or module.startswith(f"{entry}.") for entry in allowed)


def violations(package_directory: Path, allowed: frozenset[str]) -> list[str]:
    problems = []
    for path in sorted(package_directory.rglob("*.py")):
        for module in sorted(imported_modules(path.read_text(encoding="utf-8"))):
            if not is_allowed(module, allowed):
                location = path.relative_to(package_directory).as_posix()
                problems.append(f"{package_directory.name}/{location} imports {module}")
    return problems


# import-linter configures logging with ``disable_existing_loggers=True``, which would
# silently disable every logger created by earlier tests. It therefore runs in its own
# interpreter rather than in the test process.
_RUN_CONTRACTS = (
    "import sys; from importlinter import cli; "
    "sys.exit(cli.lint_imports(config_filename=sys.argv[1], no_cache=True, no_logo=True))"
)


def test_import_contracts_hold() -> None:
    result = subprocess.run(  # noqa: S603 - this interpreter and fixed arguments, no shell
        [sys.executable, "-c", _RUN_CONTRACTS, str(REPOSITORY / "pyproject.toml")],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_packages_import_only_what_the_architecture_allows(package: str) -> None:
    assert violations(PACKAGE / package, ALLOWED[package]) == []


# --- the scanner itself ------------------------------------------------------


def test_the_scanner_finds_every_form_of_import() -> None:
    source = (
        "import os\n"
        "import numpy as np, requests\n"
        "from redis import asyncio\n"
        "from . import sibling\n"
        "def late():\n"
        "    import sqlalchemy\n"
    )
    assert imported_modules(source) == {"os", "numpy", "requests", "redis", "sqlalchemy"}


def test_the_scanner_flags_unlisted_dependencies_and_forbidden_layers(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("import json\nimport numpy\nfrom civbox.core import config\n")
    (tmp_path / "bad.py").write_text(
        "import json\nimport requests\nfrom civbox.api import app\ndef f():\n    import redis\n"
    )
    sim = ALLOWED["sim"]
    assert violations(tmp_path, sim) == [
        f"{tmp_path.name}/bad.py imports civbox.api",
        f"{tmp_path.name}/bad.py imports redis",
        f"{tmp_path.name}/bad.py imports requests",
    ]


def test_allowed_prefixes_do_not_match_lookalike_names() -> None:
    allowed = frozenset({"civbox.core", "numpy"})
    assert is_allowed("civbox.core.logging", allowed)
    assert is_allowed("numpy.random", allowed)
    assert not is_allowed("civbox.coreplus", allowed)
    assert not is_allowed("numpyish", allowed)

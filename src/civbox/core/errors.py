"""Application error types.

Errors raised on purpose derive from :class:`CivboxError`. Each carries a stable
machine-readable ``code`` so callers (API envelope, worker, CLI) can react to the
category without parsing messages. Messages must never embed secret values.
"""

from collections.abc import Sequence
from typing import ClassVar


class CivboxError(Exception):
    """Base class for errors the application raises deliberately."""

    code: ClassVar[str] = "internal_error"


class ConfigurationError(CivboxError):
    """The process configuration is invalid.

    ``problems`` holds ``(setting_name, reason)`` pairs. The offending input value is
    deliberately never recorded, because it may be a credential.
    """

    code: ClassVar[str] = "configuration_error"

    def __init__(self, problems: Sequence[tuple[str, str]]) -> None:
        self.problems: tuple[tuple[str, str], ...] = tuple(problems)
        super().__init__("; ".join(f"{name}: {reason}" for name, reason in self.problems))

"""``civbox`` command-line interface.

Exit codes: 0 success, 2 usage error (argparse), 3 invalid configuration.
"""

import argparse
import json
import sys
from collections.abc import Sequence

from civbox import __version__
from civbox.core.config import load_settings, settings_for_display
from civbox.core.errors import ConfigurationError
from civbox.core.redaction import redact_string

EXIT_OK = 0
EXIT_CONFIGURATION_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="civbox", description="Civilization in a Box")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    commands.add_parser("version", help="print the installed version")

    config = commands.add_parser("config", help="inspect process configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True, metavar="ACTION")
    config_commands.add_parser("show", help="print the effective settings, secrets redacted")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "version":
        print(__version__)
        return EXIT_OK

    # The parser requires a subcommand and only defines "version" and "config".
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        print(f"configuration error: {redact_string(str(exc))}", file=sys.stderr)
        return EXIT_CONFIGURATION_ERROR

    print(json.dumps(settings_for_display(settings), indent=2, sort_keys=True))
    return EXIT_OK

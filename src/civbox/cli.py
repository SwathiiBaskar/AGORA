"""``civbox`` command-line interface.

Exit codes: 0 success, 2 usage error (argparse), 3 invalid configuration.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from civbox import __version__
from civbox.core.config import load_settings, settings_for_display
from civbox.core.errors import ConfigurationError
from civbox.core.redaction import redact_string

EXIT_OK = 0
EXIT_CONFIGURATION_ERROR = 3

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a port number: {value!r}") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port must be between 1 and 65535, got {port}")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="civbox", description="Civilization in a Box")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    commands.add_parser("version", help="print the installed version")

    config = commands.add_parser("config", help="inspect process configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True, metavar="ACTION")
    config_commands.add_parser("show", help="print the effective settings, secrets redacted")

    serve = commands.add_parser("serve", help="run the public HTTP API")
    serve.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"interface to bind (default {DEFAULT_HOST}; use 0.0.0.0 inside a container)",
    )
    serve.add_argument("--port", type=_port, default=DEFAULT_PORT, help="port (default 8000)")

    openapi = commands.add_parser("openapi", help="print the public API's OpenAPI document")
    openapi.add_argument(
        "--output",
        type=Path,
        help="write the document to this file (UTF-8, LF line endings) instead of stdout",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "version":
        print(__version__)
        return EXIT_OK

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        print(f"configuration error: {redact_string(str(exc))}", file=sys.stderr)
        return EXIT_CONFIGURATION_ERROR

    if args.command == "config":
        print(json.dumps(settings_for_display(settings), indent=2, sort_keys=True))
    elif args.command == "serve":
        # Imported here so that cheap commands do not pay for the web stack.
        from civbox.api.server import serve

        serve(settings, host=args.host, port=args.port)
    else:
        from civbox.api.app import create_app

        document = json.dumps(
            create_app(settings).openapi(), indent=2, sort_keys=True, ensure_ascii=False
        )
        if args.output is None:
            print(document)
        else:
            # Explicit encoding and newline: shell redirection on Windows would
            # otherwise re-encode the document and change its line endings.
            args.output.write_text(document + "\n", encoding="utf-8", newline="\n")
    return EXIT_OK

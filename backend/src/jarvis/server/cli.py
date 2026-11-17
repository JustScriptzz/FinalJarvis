"""Command line entry point: ``jarvis serve``."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from ..core.config import ConfigError, load_config

BANNER = r"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
      Just A Rather Very Intelligent System
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="J.A.R.V.I.S. — local-first personal AI",
    )
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start the web interface")
    serve.add_argument("--config", default=None, help="path to config.yaml")
    serve.add_argument("--host", default=None, help="bind address (default from config)")
    serve.add_argument("--port", type=int, default=None, help="port (default from config)")
    serve.add_argument("--reload", action="store_true", help="reload on source changes")
    serve.add_argument("--log-level", default=None, help="uvicorn log level")
    serve.add_argument("--open", action="store_true", help="open the dashboard in a browser")

    check = sub.add_parser("config-check", help="validate config and print a redacted summary")
    check.add_argument("--config", default=None)

    sub.add_parser("version", help="print the version")
    return parser


def _resolve_config(path: str | None) -> Path | None:
    if path:
        return Path(path).expanduser().resolve()
    env = os.environ.get("JARVIS_CONFIG")
    return Path(env).expanduser().resolve() if env else None


def _serve(args: argparse.Namespace) -> int:
    config_path = _resolve_config(args.config)
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"JARVIS configuration error:\n{exc}", file=sys.stderr)
        return 2

    host = args.host or config.server.host
    port = args.port or config.server.port
    log_level = (args.log_level or config.app.log_level).lower()
    url = f"http://{host if host not in {'0.0.0.0', '::'} else 'localhost'}:{port}"

    print(BANNER)
    print(f"  interface   {url}")
    print(f"  environment {config.app.environment}")
    print(f"  data        {config.app.data_dir}")
    print(f"  models      llm={config.models.llm.active}  "
          f"stt={config.models.stt.active}  tts={config.models.tts.active}")
    print("  core selection is completed in the browser.\n")

    if config_path is not None:
        os.environ["JARVIS_CONFIG"] = str(config_path)

    import uvicorn  # imported late so --help stays fast

    if args.reload:
        reload_dir = Path(__file__).resolve().parents[1]
        uvicorn.run(
            "jarvis.server.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=[str(reload_dir)],
            log_level=log_level,
            access_log=False,
        )
        return 0

    from .app import create_app

    app = create_app(config_path)
    if args.open:
        import threading
        import webbrowser

        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level=log_level, access_log=False)
    return 0


def _config_check(args: argparse.Namespace) -> int:
    try:
        config = load_config(_resolve_config(args.config))
    except ConfigError as exc:
        print(f"Configuration invalid:\n{exc}", file=sys.stderr)
        return 2
    import json

    print(json.dumps(config.summary(), indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    command = args.command or "serve"
    if command == "serve":
        if not hasattr(args, "config"):
            args = parser.parse_args(["serve", *(argv or [])])
        return _serve(args)
    if command == "config-check":
        return _config_check(args)
    if command == "version":
        from .. import __version__
        from .app import create_app  # noqa: F401  (keeps package import side effects minimal)

        print(f"JARVIS {__version__}")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

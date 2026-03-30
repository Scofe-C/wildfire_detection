"""run_dashboard.py — Start the OBJ-3 operator dashboard.

Usage
-----
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 8080 --no-browser

What it does
------------
1. Checks Ollama is reachable (warning only if not)
2. Starts the FastAPI server via uvicorn
3. Opens the dashboard in your default browser
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import webbrowser
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_dashboard")


def check_ollama(base_url: str, model: str) -> bool:
    """Return True if Ollama is reachable and model is available."""
    try:
        import ollama as ollama_lib  # noqa
        client = ollama_lib.Client(host=base_url)
        available = [m.model for m in client.list().models]
        found = any(
            name == model or name == f"{model}:latest"
            for name in available
        )
        if found:
            logger.info("Ollama ✅  model=%s available", model)
        else:
            logger.warning(
                "Ollama is running but model '%s' is not pulled. "
                "Run: ollama pull %s", model, model,
            )
        return found
    except Exception as exc:
        logger.warning(
            "Ollama not reachable at %s: %s\n"
            "  → Start with: ollama serve", base_url, exc,
        )
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the OBJ-3 dashboard.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--reload", action="store_true", help="Hot-reload on code changes (dev mode)")
    args = parser.parse_args()

    # --- Check config ---
    try:
        import yaml  # noqa
        config_path = _ROOT / "configs" / "reporting_config.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        backend = cfg.get("llm_backend", "ollama")
        logger.info("Active backend: %s", backend)

        if backend == "ollama":
            ollama_cfg = cfg.get("ollama", {})
            check_ollama(
                base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
                model=ollama_cfg.get("model", "qwen3:8b"),
            )
        elif backend == "gemini_dev":
            import os  # noqa
            if not os.getenv("GEMINI_API_KEY"):
                logger.warning("GEMINI_API_KEY env var not set — Gemini backend will fail")
            else:
                logger.info("Gemini Dev API ✅  key configured")
    except Exception as exc:
        logger.warning("Config check failed: %s", exc)

    # --- Start server ---
    url = f"http://{args.host}:{args.port}"
    logger.info("=" * 55)
    logger.info("  OBJ-3 Dashboard starting at %s", url)
    logger.info("  Press Ctrl+C to stop")
    logger.info("=" * 55)

    # Open browser after a short delay (server needs a moment to start)
    if not args.no_browser:
        import threading  # noqa
        def _open():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        import uvicorn  # noqa
    except ImportError:
        logger.error(
            "uvicorn not installed. Install with:\n"
            "  pip install uvicorn fastapi python-multipart"
        )
        return 1

    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

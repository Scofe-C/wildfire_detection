"""run_dashboard.py — Start the OBJ-3 operator dashboard.

Usage
-----
    python scripts/run_dashboard.py --port 8080 --no-browser

What it does
------------
1. Validates config from reporting_config.yaml
2. Checks LLM backend availability (Ollama is optional — Gemini API is sufficient)
3. Starts the FastAPI server via uvicorn
4. Opens the dashboard in your default browser

Cross-platform: macOS, Linux, Windows.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
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
    """Return True if Ollama is reachable. Non-blocking — failure is OK."""
    try:
        import ollama as ollama_lib
        client = ollama_lib.Client(host=base_url)
        available = [m.model for m in client.list().models]
        found = any(
            name == model or name == f"{model}:latest"
            for name in available
        )
        if found:
            logger.info("Ollama OK — model=%s available", model)
        else:
            logger.info(
                "Ollama running but model '%s' not pulled. "
                "Run: ollama pull %s  (optional — Gemini API works without Ollama)",
                model, model,
            )
        return found
    except ImportError:
        logger.info("Ollama package not installed — skipping (Gemini API is sufficient)")
        return False
    except Exception:
        logger.info("Ollama not reachable at %s — skipping (Gemini API is sufficient)", base_url)
        return False


def check_gemini() -> bool:
    """Check if Gemini API key is configured."""
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        logger.info("Gemini Dev API OK — key configured")
        return True
    else:
        logger.warning(
            "GEMINI_API_KEY not set. Set it with:\n"
            "  export GEMINI_API_KEY='your-key-here'   # macOS/Linux\n"
            "  set GEMINI_API_KEY=your-key-here         # Windows cmd\n"
            "  $env:GEMINI_API_KEY='your-key-here'      # Windows PowerShell"
        )
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the OBJ-3 dashboard.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--reload", action="store_true", help="Hot-reload on code changes (dev mode)")
    args = parser.parse_args()

    logger.info("Platform: %s %s", platform.system(), platform.machine())

    # --- Check config ---
    try:
        import yaml
        config_path = _ROOT / "configs" / "reporting_config.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        backend = cfg.get("llm_backend", "gemini_dev")
        logger.info("Active LLM backend: %s", backend)

        if backend == "ollama":
            ollama_cfg = cfg.get("ollama", {})
            ok = check_ollama(
                base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
                model=ollama_cfg.get("model", "qwen3:8b"),
            )
            if not ok:
                logger.info(
                    "Tip: Switch to Gemini API by setting llm_backend: 'gemini_dev' "
                    "in configs/reporting_config.yaml"
                )
        elif backend == "gemini_dev":
            check_gemini()
        elif backend == "vertex_ai":
            logger.info("Vertex AI backend — requires GCP credentials")
        else:
            logger.warning("Unknown backend: %s", backend)
    except FileNotFoundError:
        logger.warning("Config not found: configs/reporting_config.yaml — using defaults")
    except Exception as exc:
        logger.warning("Config check failed: %s", exc)

    # --- Start server ---
    url = f"http://{args.host}:{args.port}"
    logger.info("=" * 55)
    logger.info("  OBJ-3 Dashboard starting at %s", url)
    logger.info("  Press Ctrl+C to stop")
    logger.info("=" * 55)

    # Open browser after a short delay
    if not args.no_browser:
        import threading

        def _open():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        import uvicorn
    except ImportError:
        logger.error(
            "uvicorn not installed. Install with:\n"
            "  pip install uvicorn fastapi python-multipart\n"
            "Or install all dependencies:\n"
            "  pip install -r requirements.txt"
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

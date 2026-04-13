
"""List available Gemini models and their capabilities.

Usage:
    cd model-pipeline
    python scripts/list_gemini_models.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY env var not set.")
        print("Get a free key from https://aistudio.google.com/apikey")
        sys.exit(1)

    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai not installed. Run: uv pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print("=" * 70)
    print("Available Gemini Models (generateContent supported)")
    print("=" * 70)

    models = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
        if "generateContent" in actions:
            models.append(m)

    # Sort by name for readability
    models.sort(key=lambda m: m.name)

    for m in models:
        display = getattr(m, "display_name", "")
        desc = (getattr(m, "description", "") or "")[:80]
        input_limit = getattr(m, "input_token_limit", "?")
        output_limit = getattr(m, "output_token_limit", "?")
        print(f"\n  Model:   {m.name}")
        print(f"  Display: {display}")
        print(f"  Tokens:  {input_limit} in / {output_limit} out")
        if desc:
            print(f"  Desc:    {desc}")

    print(f"\n{'=' * 70}")
    print(f"Total: {len(models)} models")
    print(f"{'=' * 70}")

    # Check which of our priority models are available
    priority = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    available_names = {m.name for m in models}

    print("\nFallback priority status:")
    for p in priority:
        target = f"models/{p}"
        status = "AVAILABLE" if target in available_names else "NOT FOUND"
        print(f"  {p:30s} {status}")


if __name__ == "__main__":
    main()
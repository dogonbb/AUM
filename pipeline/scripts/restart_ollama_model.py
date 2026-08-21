"""Unload one Ollama model so the next request loads a fresh instance."""

from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    args = parser.parse_args()
    endpoint = args.base_url.rstrip("/") + "/api/generate"
    payload = {"model": args.model, "keep_alive": 0, "stream": False}
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        response.read()
    print(f"Unloaded Ollama model: {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

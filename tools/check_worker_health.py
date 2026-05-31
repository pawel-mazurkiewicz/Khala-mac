"""Poll a worker's /health until it reports a target status, or time out.
Run: .venv-mac/bin/python tools/check_worker_health.py <url> [target] [timeout_s]
Prints the final status JSON line and exits 0 if target reached, 1 otherwise."""
from __future__ import annotations

import json
import sys
import time
import urllib.request


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011/health"
    target = sys.argv[2] if len(sys.argv) > 2 else "idle"
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                last = json.loads(r.read().decode())
                if last.get("status") == target:
                    print(json.dumps(last))
                    return 0
        except Exception as exc:  # noqa: BLE001
            last = {"error": str(exc)}
        time.sleep(3)
    print(json.dumps(last or {"error": "no response"}))
    return 1


if __name__ == "__main__":
    sys.exit(main())

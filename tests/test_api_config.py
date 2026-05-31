"""Phase-UI: NUM_TRACKS_PER_JOB must be env-overridable (default 2, Mac sets 1).
Run: .venv-mac/bin/python tests/test_api_config.py"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = str(Path(__file__).resolve().parent.parent / "backend")
PY = sys.executable


def _read_tracks(env_val: str | None) -> int:
    import os
    env = dict(os.environ)
    env.pop("KHALA_TRACKS_PER_JOB", None)
    if env_val is not None:
        env["KHALA_TRACKS_PER_JOB"] = env_val
    out = subprocess.check_output(
        [PY, "-c", "import sys; sys.path.insert(0, r'%s'); "
                   "import backend_api; print(backend_api.NUM_TRACKS_PER_JOB)" % BACKEND],
        env=env, text=True,
    ).strip()
    return int(out.splitlines()[-1])


def test_tracks_per_job_env():
    assert _read_tracks(None) == 2, "default must stay 2 (CUDA behavior unchanged)"
    assert _read_tracks("1") == 1, "KHALA_TRACKS_PER_JOB=1 must override to 1"
    print("  test_tracks_per_job_env PASS (default=2, override=1)")


if __name__ == "__main__":
    test_tracks_per_job_env()
    print("ALL PASS")

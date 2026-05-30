#!/usr/bin/env bash
# scripts/cuda/remote.sh — Mac → CUDA bisection harness.
#
# Pushes .npz fixtures to a vast.ai rental, runs cumesh / cuda-side
# diagnostics in /venv/pixal3d, fetches result artefacts back. All test
# scripts emit a single JSON line on the last line of stdout so callers
# can grep '^{' to pick it up.
#
# Usage:
#   ./remote.sh bootstrap               sync run_*.py, verify imports
#   ./remote.sh push <local-file>       rsync into remote in/
#   ./remote.sh pull <pattern> [dest]   rsync from remote out/ to dest (default /tmp/cuda_results/)
#   ./remote.sh run <test> [args...]    execute run_<test>.py on remote
#   ./remote.sh shell                   interactive ssh
#   ./remote.sh raw <cmd...>            arbitrary remote command (joined as one shell string)
#
# Convention: positional file args to run_*.py are basenames resolved
# inside $CUDA_REMOTE_DIR/in/; abs paths are passed through unchanged.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVF="$ROOT/.env"
if [[ ! -f "$ENVF" ]]; then
  echo "missing $ENVF — copy from .env.example and edit" >&2
  exit 2
fi
# shellcheck disable=SC1090
. "$ENVF"
: "${CUDA_HOST:?}" "${CUDA_PORT:?}" "${CUDA_USER:?}" "${CUDA_PY:?}" "${CUDA_REMOTE_DIR:?}"

SSH_OPTS=(-p "$CUDA_PORT" -o StrictHostKeyChecking=accept-new)
ssh_cmd() { ssh "${SSH_OPTS[@]}" "$CUDA_USER@$CUDA_HOST" "$@"; }
rsync_tx() { rsync -e "ssh -p $CUDA_PORT -o StrictHostKeyChecking=accept-new" -ahz --progress "$@"; }

cmd=${1:-help}; shift || true

case "$cmd" in
  bootstrap)
    ssh_cmd "mkdir -p '$CUDA_REMOTE_DIR/in' '$CUDA_REMOTE_DIR/out'"
    # Sync only the run_*.py scripts (skip .env*, this wrapper)
    rsync_tx \
      "$ROOT"/run_*.py \
      "$CUDA_USER@$CUDA_HOST:$CUDA_REMOTE_DIR/"
    ssh_cmd "$CUDA_PY -c 'import sys, cumesh, trimesh, numpy, torch; print(\"bootstrap OK\"); print(\"  python:\", sys.executable); print(\"  cumesh:\", cumesh.__file__); print(\"  trimesh:\", trimesh.__version__); print(\"  numpy:\", numpy.__version__); print(\"  torch:\", torch.__version__, \"cuda=\", torch.cuda.is_available())'"
    ;;

  push)
    src=${1:?usage: push <local-file>}
    [[ -f "$src" ]] || { echo "no such file: $src" >&2; exit 1; }
    rsync_tx "$src" "$CUDA_USER@$CUDA_HOST:$CUDA_REMOTE_DIR/in/"
    echo "pushed $(basename "$src") → $CUDA_REMOTE_DIR/in/"
    ;;

  pull)
    pat=${1:?usage: pull <remote-pattern> [dest]}
    dest=${2:-/tmp/cuda_results/}
    mkdir -p "$dest"
    rsync_tx "$CUDA_USER@$CUDA_HOST:$CUDA_REMOTE_DIR/out/$pat" "$dest"
    echo "pulled $pat → $dest"
    ;;

  run)
    test=${1:?usage: run <test> [args...]}; shift || true
    # quote each remaining arg for the remote shell
    remote_args=""
    for a in "$@"; do
      printf -v q '%q' "$a"
      remote_args+=" $q"
    done
    ssh_cmd "cd '$CUDA_REMOTE_DIR' && $CUDA_PY run_${test}.py${remote_args}"
    ;;

  shell)
    exec ssh "${SSH_OPTS[@]}" "$CUDA_USER@$CUDA_HOST"
    ;;

  raw)
    ssh_cmd "$*"
    ;;

  help|--help|-h|"")
    sed -n '2,/^set -euo/p' "$0" | sed -n 's/^# \{0,1\}//p' | sed '$d'
    ;;

  *)
    echo "unknown command: $cmd (try: bootstrap | push | pull | run | shell | raw | help)" >&2
    exit 2
    ;;
esac

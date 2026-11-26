#!/usr/bin/env bash
# Starts STEM OS on a remote GPU host such as a RunPod pod.
# Both servers listen on all interfaces so the host's HTTPS proxy can reach them:
#   frontend  https://<pod-id>-5173.proxy.runpod.net/
#   backend   https://<pod-id>-8000.proxy.runpod.net/   (e.g. /api/health)
# The frontend forwards /api to the backend itself, so the browser only needs the frontend address.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"

(cd backend && exec "$PYTHON" -m uvicorn app.main:app --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT INT TERM

cd frontend
[ -d node_modules ] || npm install
npm run dev -- --host 0.0.0.0 --port 5173 --strictPort

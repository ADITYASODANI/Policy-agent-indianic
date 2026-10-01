#!/usr/bin/env bash
# Start MongoDB (Docker), the FastAPI backend and the React frontend together.
# Ctrl+C stops the backend and frontend; MongoDB keeps running.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

# --- MongoDB -------------------------------------------------------------
if nc -z localhost 27017 2>/dev/null; then
  echo "✓ MongoDB already running on :27017"
elif command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  if docker ps -a --format '{{.Names}}' | grep -qx policy-mongo; then
    docker start policy-mongo >/dev/null
  else
    docker run -d --name policy-mongo -p 27017:27017 mongo:7 >/dev/null
  fi
  echo "✓ MongoDB started (Docker container: policy-mongo)"
else
  echo "! MongoDB is not running and Docker is unavailable. Chat will work, but history won't be saved."
fi

# --- Backend -------------------------------------------------------------
if [ ! -x "$BACKEND/.venv/bin/python" ]; then
  echo "✗ backend/.venv not found. Run the backend setup in README.md first." >&2
  exit 1
fi
if [ ! -f "$BACKEND/.env" ]; then
  echo "✗ backend/.env not found. Copy backend/.env.example and set GROQ_API_KEY." >&2
  exit 1
fi
if [ ! -d "$FRONTEND/node_modules" ]; then
  echo "• Installing frontend packages…"
  (cd "$FRONTEND" && npm install)
fi

cd "$BACKEND"
if [ ! -d chroma_db ]; then
  echo "• Policy not ingested yet, running ingestion…"
  .venv/bin/python -m scripts.ingest
fi

cleanup() {
  echo; echo "Stopping…"
  kill "${API_PID:-}" "${UI_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

.venv/bin/uvicorn app.main:app --reload --reload-dir app --port 8000 &
API_PID=$!

cd "$FRONTEND"
npm run dev -- --port 5173 &
UI_PID=$!

echo
echo "  UI:       http://localhost:5173"
echo "  API docs: http://localhost:8000/docs"
echo "  Press Ctrl+C to stop."
wait

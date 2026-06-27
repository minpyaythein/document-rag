#!/usr/bin/env bash
#
# local_setup.sh — sync dependencies and launch the DocumentRAG app locally.
#
# Usage:
#   ./local_setup.sh
#
set -euo pipefail

# Run from the script's own directory so it works no matter where it's called from.
cd "$(dirname "$0")"

# uv is the package manager (see pyproject.toml / README).
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is not installed. See https://docs.astral.sh/uv/ for install steps." >&2
  exit 1
fi

# The app needs GOOGLE_API_KEY, ZAI_API_KEY, and ZAI_MODEL — usually from .env.
if [[ ! -f .env ]]; then
  echo "Warning: no .env found. The app needs GOOGLE_API_KEY, ZAI_API_KEY, and ZAI_MODEL." >&2
fi

# Make sure the virtualenv matches the lockfile, then launch.
echo "Syncing dependencies..."
uv sync

echo "Starting Streamlit on http://localhost:8501 ..."
exec uv run streamlit run main.py

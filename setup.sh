#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Setting up evaluation environment..."
if ! "${SCRIPT_DIR}/venv/bin/python" - <<'PY' >/dev/null 2>&1
import openai
PY
then
    echo "Installing Python dependencies into bundled venv..."
    "${SCRIPT_DIR}/venv/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt"
fi
if [ ! -f "${SCRIPT_DIR}/.env" ]; then
    cp "${SCRIPT_DIR}/.env.example" "${SCRIPT_DIR}/.env"
    echo "Created .env - please edit with your API keys"
fi
echo "Setup complete. Edit .env then run: bash run_all.sh"

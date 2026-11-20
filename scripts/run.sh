#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${SPHWEB_VENV:-$HOME/.venvs/sphweb}"
PORT="${SPHWEB_PORT:-8766}"

. "$VENV/bin/activate"
cd "$ROOT"
export PYTHONUTF8=1
export NO_PROXY="${NO_PROXY:-${no_proxy:-}},localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
if [ -n "${SPHWEB_CA_BUNDLE:-}" ]; then
  export REQUESTS_CA_BUNDLE="$SPHWEB_CA_BUNDLE" SSL_CERT_FILE="$SPHWEB_CA_BUNDLE"
fi

URL="http://localhost:$PORT"
(
  sleep 2
  if command -v wslview >/dev/null 2>&1; then wslview "$URL" >/dev/null 2>&1 || true
  elif command -v cmd.exe >/dev/null 2>&1; then cmd.exe /c start "" "$URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then open "$URL" || true
  fi
) &
echo "브라우저에서 $URL 을 여세요"
exec python -m sphweb --host 127.0.0.1 --port "$PORT" --no-browser

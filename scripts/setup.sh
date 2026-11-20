#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${SPHWEB_VENV:-$HOME/.venvs/sphweb}"
PY="${SPHWEB_PYTHON:-python3}"
CORE="${SIMWEB_CORE:-$ROOT/../Simweb-Core}"

"$PY" -c 'import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)' || { echo "Python 3.10 ~ 3.13 이 필요합니다 (taichi 지원 범위)"; exit 1; }

if [ -n "${SPHWEB_CA_BUNDLE:-}" ]; then
  export PIP_CERT="$SPHWEB_CA_BUNDLE" REQUESTS_CA_BUNDLE="$SPHWEB_CA_BUNDLE" SSL_CERT_FILE="$SPHWEB_CA_BUNDLE"
fi

[ -d "$VENV" ] || "$PY" -m venv "$VENV"
. "$VENV/bin/activate"
python -m pip install -U pip wheel
if [ -d "$CORE" ]; then
  python -m pip install -e "$CORE"
else
  python -m pip install "simweb-core @ git+https://github.com/MunkyunLEE/Simweb-Core.git@main"
fi
python -m pip install -e "$ROOT[test]"
python "$ROOT/examples/make_examples.py"
python -m sphweb.engine.backend --bench 2000000

echo
echo "설치 완료. 실행: $ROOT/scripts/run.sh"

#!/usr/bin/env bash
# Deploy de la Lambda persist_event (src/cloud/persist_event.py).
#
# Usage:
#   scripts/deploy_lambda.sh [environment]
#
# Default: environment=dev. Requiere awscli con write access a la Lambda.
# Nombre de la función matchea el resource CloudFormation:
#   people-counter-persist-event-<environment>
#
# La Lambda necesita psycopg como dependencia. Para PoC empaquetamos psycopg
# binary wheel en el zip (~10 MB). Para prod considerar una Lambda Layer
# compartida (scripts/build_psycopg_layer.sh — TODO).

set -euo pipefail

ENV="${1:-dev}"
FUNCTION_NAME="people-counter-persist-event-${ENV}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_FILE="${REPO_ROOT}/src/cloud/persist_event.py"

if [[ ! -f "${SRC_FILE}" ]]; then
  echo "ERROR: ${SRC_FILE} not found" >&2
  exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

# Copia el código fuente.
cp "${SRC_FILE}" "${TMPDIR}/persist_event.py"

# Instala psycopg (binary) target el runtime de Lambda (manylinux2014).
# python3.13 en Lambda corre x86_64 por default (arm64 con configuración).
pip install \
  --platform manylinux2014_x86_64 \
  --target "${TMPDIR}" \
  --implementation cp \
  --python-version 3.13 \
  --only-binary=:all: \
  --upgrade \
  "psycopg[binary]==3.2.*" >/dev/null

# Empaqueta todo en zip plano.
( cd "${TMPDIR}" && zip -qr lambda.zip persist_event.py ./psycopg* ./psycopg_binary* ./pyproject.toml 2>/dev/null || true )
( cd "${TMPDIR}" && zip -qr lambda.zip . -x '*.dist-info/*' )

SIZE=$(stat -c%s "${TMPDIR}/lambda.zip" 2>/dev/null || stat -f%z "${TMPDIR}/lambda.zip")
echo "Package size: $((SIZE / 1024)) KB"

echo "Deploying ${FUNCTION_NAME}..."
aws lambda update-function-code \
  --function-name "${FUNCTION_NAME}" \
  --zip-file "fileb://${TMPDIR}/lambda.zip" \
  --publish \
  --output table \
  --query '{Function:FunctionName,Version:Version,LastModified:LastModified,Size:CodeSize}'

echo "Done."

#!/usr/bin/env bash
# Deploy de las Lambdas de cloud (src/cloud/*.py).
#
# Usage:
#   scripts/deploy_lambda.sh [environment] [function]
#   function: persist_event (default) | query_aggregates | ingest_pos_transaction
#
# Default: environment=dev, function=persist_event. Requiere awscli con write
# access a la Lambda. Cada CFN update resetea las 3 Lambdas al ZipFile
# placeholder → hay que redeployar las tres con este script.
#
# La Lambda necesita psycopg como dependencia. Para PoC empaquetamos psycopg
# binary wheel en el zip (~10 MB). Para prod considerar una Lambda Layer
# compartida (scripts/build_psycopg_layer.sh — TODO).

set -euo pipefail

ENV="${1:-dev}"
FUNCTION="${2:-persist_event}"

case "${FUNCTION}" in
  persist_event)          FUNCTION_NAME="people-counter-persist-event-${ENV}";   SRC_REL="src/cloud/persist_event.py" ;;
  query_aggregates)       FUNCTION_NAME="people-counter-query-aggregates-${ENV}"; SRC_REL="src/cloud/query_aggregates.py" ;;
  ingest_pos_transaction) FUNCTION_NAME="people-counter-ingest-pos-${ENV}";       SRC_REL="src/cloud/ingest_pos_transaction.py" ;;
  *) echo "ERROR: función desconocida: ${FUNCTION} (persist_event|query_aggregates|ingest_pos_transaction)" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_FILE="${REPO_ROOT}/${SRC_REL}"

if [[ ! -f "${SRC_FILE}" ]]; then
  echo "ERROR: ${SRC_FILE} not found" >&2
  exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

# Copia el código fuente (el handler es <modulo>.handler → nombre original).
cp "${SRC_FILE}" "${TMPDIR}/$(basename "${SRC_FILE}")"

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

# Empaqueta todo en zip plano (el .py del handler + psycopg).
( cd "${TMPDIR}" && zip -qr lambda.zip . -x '*.dist-info/*' )

SIZE=$(stat -c%s "${TMPDIR}/lambda.zip" 2>/dev/null || stat -f%z "${TMPDIR}/lambda.zip")
echo "Package size: $((SIZE / 1024)) KB"

echo "Deploying ${FUNCTION_NAME}..."
# NO usamos --publish para evitar que se acumulen Lambda versions
# numeradas a cada deploy. Para PoC con 1 device la IoT Rule invoca
# $LATEST directamente, no hace falta versioning.
aws lambda update-function-code \
  --function-name "${FUNCTION_NAME}" \
  --zip-file "fileb://${TMPDIR}/lambda.zip" \
  --output table \
  --query '{Function:FunctionName,LastModified:LastModified,Size:CodeSize}'

echo "Done."

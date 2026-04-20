#!/usr/bin/env bash
# Deploy the dedup L3 Lambda from src/cloud/lambda_dedup.py.
#
# Usage:
#   scripts/deploy_lambda.sh [environment]
#
# Defaults to environment=prod. Requires awscli configured with write access
# to the target Lambda function. Function name must match the CloudFormation
# resource: people-counter-dedup-<environment>.

set -euo pipefail

ENV="${1:-prod}"
FUNCTION_NAME="people-counter-dedup-${ENV}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_FILE="${REPO_ROOT}/src/cloud/lambda_dedup.py"

if [[ ! -f "${SRC_FILE}" ]]; then
  echo "ERROR: ${SRC_FILE} not found" >&2
  exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

cp "${SRC_FILE}" "${TMPDIR}/lambda_dedup.py"
( cd "${TMPDIR}" && zip -q lambda.zip lambda_dedup.py )

echo "Deploying ${FUNCTION_NAME}..."
aws lambda update-function-code \
  --function-name "${FUNCTION_NAME}" \
  --zip-file "fileb://${TMPDIR}/lambda.zip" \
  --publish \
  --output table \
  --query '{Function:FunctionName,Version:Version,LastModified:LastModified,Size:CodeSize}'

echo "Done."

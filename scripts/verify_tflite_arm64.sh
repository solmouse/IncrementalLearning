#!/bin/bash
set -e

echo '=== 1. Install Basic Tools ==='
apt-get update
apt-get install -y build-essential file

echo '=== 2. Define Paths and Verify Files ==='
SDK_DIR="/workspace/sdk"
TEST_DIR="${SDK_DIR}/tests"
MODEL_DIR="${SDK_DIR}/models"
SOURCE_FILE="${TEST_DIR}/phase2_with_flex.cpp" 
OUTPUT_BINARY="${TEST_DIR}/phase2_arm64"

# 파일 존재 여부 확인
echo "Verifying model and source files exist..."
test -f "${SOURCE_FILE}" || { echo "❌ ERROR: Source file not found: ${SOURCE_FILE}"; exit 1; }
test -f "${MODEL_DIR}/model.tflite" || { echo "❌ ERROR: Model file not found: ${MODEL_DIR}/model.tflite"; exit 1; }
echo "✅ All required files found."

echo '=== 3. Compile ARM64 Binary ==='

g++ ${SOURCE_FILE} \
    -I${SDK_DIR}/include \
    -L${SDK_DIR}/lib \
    -ltensorflowlite -ltensorflowlite_flex \
    -lpthread -ldl -lm \
    -std=c++17 \
    -Wl,-rpath,'$ORIGIN/../lib' \
    -o ${OUTPUT_BINARY}

echo "✓ ARM64 Binary created at: ${OUTPUT_BINARY}"

# 바이너리 아키텍처 확인
echo "Verifying binary architecture:"
file ${OUTPUT_BINARY}

echo '=== 4. Execute ARM64 Binary ==='

cd ${TEST_DIR}

export LD_LIBRARY_PATH="${SDK_DIR}/lib:$LD_LIBRARY_PATH"

echo "Running command: ./${OUTPUT_BINARY##*/} ../models/model.tflite ../models/ckpt_before.npy ../models/ckpt_after.npy"
./${OUTPUT_BINARY##*/} ../models/model.tflite ../models/ckpt_before.npy ../models/ckpt_after.npy

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo '=== ✓ Verification Success (Exit Code 0) ==='
else
    echo "=== ❌ Verification FAILED: Execution failed with exit code ${EXIT_CODE}. ==="
    exit 1
fi
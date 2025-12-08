#!/bin/bash
set -e

echo "=========================================="
echo "  Raspberry Pi - TFLite Setup & Run"
echo "=========================================="

# 1. 현재 위치 확인
if [ ! -f "tflite_sdk_arm64.tar.gz" ]; then
    echo "X ERROR: tflite_sdk_arm64.tar.gz not found!"
    echo "Please run this script from pi/cpp_pi_test_251208/ directory"
    exit 1
fi

# 2. 압축 해제
echo ""
echo "=== Extracting SDK ==="
tar -xzvf tflite_sdk_arm64.tar.gz
cd sdk

# 3. glibc 버전 확인
echo ""
echo "=== Checking glibc version ==="
GLIBC_VER=$(ldd --version | head -n1 | awk '{print $NF}')
echo "glibc version: $GLIBC_VER (required: 2.28+)"

# 4. 파일 확인
echo ""
echo "=== Verifying files ==="
echo "Libraries:"
ls -lh lib/*.so
echo ""
echo "Source code:"
ls -lh tests/phase2_with_flex.cpp
echo ""
echo "Models:"
ls -lh models/

# 5. 컴파일
echo ""
echo "=== Compiling ==="
g++ tests/phase2_with_flex.cpp \
    -I./include \
    -L./lib \
    -ltensorflowlite -ltensorflowlite_flex \
    -lpthread -ldl -lm \
    -std=c++17 \
    -Wl,-rpath,'$ORIGIN/../lib' \
    -O3 \
    -o tests/phase2_pi

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
else
    echo "X Compilation failed!"
    exit 1
fi

# 6. 바이너리 확인
echo ""
echo "=== Binary info ==="
file tests/phase2_pi
ls -lh tests/phase2_pi

# 7. 실행
echo ""
echo "=== Running phase2 training ==="
cd tests
export LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH

./phase2_pi \
    ../models/model.tflite \
    ../models/ckpt_before.npy \
    ../models/ckpt_after.npy

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✓ SUCCESS!"
    echo "=========================================="
    echo "Output checkpoint: ../models/ckpt_after.npy"
    ls -lh ../models/ckpt_after.npy
else
    echo ""
    echo "=========================================="
    echo "  X FAILED (Exit code: $EXIT_CODE)"
    echo "=========================================="
    exit 1
fi
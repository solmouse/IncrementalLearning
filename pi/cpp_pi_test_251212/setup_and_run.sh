#!/bin/bash
set -e

echo "=========================================="
echo "  Raspberry Pi - TFLite Setup & Run"
echo "=========================================="

# 1. 현재 위치 확인
if [ ! -f "tflite_sdk_arm64.tar.gz" ]; then
    echo "X ERROR: tflite_sdk_arm64.tar.gz not found!"
    exit 1
fi

if [ ! -f "data.tar.gz" ]; then
    echo "X ERROR: data.tar.gz not found!"
    exit 1
fi

# 2-1. SDK 압축 해제
echo ""
echo "=== Extracting TFLite SDK ==="
tar -xzvf tflite_sdk_arm64.tar.gz

# 2-2. 데이터 압축 해제
echo ""
echo "=== Extracting data ==="
tar -xzvf data.tar.gz

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
echo ""
echo "Data:"
ls -lh ../data/

# 5. 컴파일 (변경)
echo ""
echo "=== Compiling ==="
g++ tests/phase2_with_flex.cpp tests/cnpy.cpp \
    -I./include \
    -L./lib \
    -ltensorflowlite \
    -lpthread -ldl -lm -lz \
    -std=c++17 -O3 \
    -Wl,-rpath,'$ORIGIN/../lib' \
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

# 7. 실행 (변경)
echo ""
echo "=== Running phase2 training ==="
cd tests
export LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH

./phase2_pi \
    ../models/model.tflite \
    ../models/ckpt_before.npy \
    ../models/ckpt_after.npy \
    ../../data/domainB_images.npy \
    ../../data/domainB_labels.npy \
    ../../data/domainA_images.npy \
    ../../data/domainA_labels.npy
    
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
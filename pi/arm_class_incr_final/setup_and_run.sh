#!/bin/bash
set -e

echo "=========================================="
echo "   Raspberry Pi - TFLite Setup"
echo "=========================================="

SDK_FILE="sdk.tar.gz"
DATA_FILE="data.tar.gz"

if [ ! -f "$SDK_FILE" ]; then
    echo "X ERROR: $SDK_FILE not found!"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "X ERROR: $DATA_FILE not found!"
    exit 1
fi

echo ""
echo "=== Extracting TFLite SDK ($SDK_FILE) ==="
tar -xzvf "$SDK_FILE"

echo ""
echo "=== Extracting Data ($DATA_FILE) ==="
tar -xzvf "$DATA_FILE"

if [ -d "sdk" ]; then
    echo "--- Moving into sdk/tests and running make ---"
    BASE_DIR=$(pwd)
    cd sdk/tests
    make run 
else
    echo "X ERROR: 'sdk' directory not found after extraction!"
    exit 1
fi

echo "=========================================="
echo "   Setup Complete!"
echo "=========================================="
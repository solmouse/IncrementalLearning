#!/bin/bash
set -e

echo '=== 1. Install Basic Tools ==='
apt-get update
apt-get install -y git wget curl build-essential python3 python3-pip python3-dev openjdk-11-jdk zip unzip

echo '=== 2. Install Bazel 6.5.0 ==='
wget https://github.com/bazelbuild/bazel/releases/download/6.5.0/bazel-6.5.0-linux-x86_64 -O /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel
bazel --version

echo '=== 3. Clone TensorFlow 2.19.0 ==='
git clone --depth 1 --branch v2.19.0 https://github.com/tensorflow/tensorflow.git /tensorflow_src
cd /tensorflow_src

echo '=== 3.5. Check Protobuf Version ==='
PROTOBUF_VER=$(grep -oP 'protobuf-\K[0-9.]+' WORKSPACE | head -1)
echo "✓ TensorFlow will use Protobuf: $PROTOBUF_VER"
if [[ "$PROTOBUF_VER" != "3.24"* ]]; then
    echo "⚠️  Note: Expected Protobuf 3.24.x but found $PROTOBUF_VER"
    echo "⚠️  Continuing with TensorFlow's default version for compatibility..."
else
    echo "✓ Using expected Protobuf 3.24.x"
fi

echo '=== 4. Python Dependencies ==='
python3 -m pip install numpy wheel --break-system-packages

echo '=== 5. Build ARM64 Shared Libraries (Official Method) ==='
echo "Building libtensorflowlite.so for ARM64..."
bazel build --config=elinux_aarch64 \
  -c opt \
  --config=monolithic \
  --define=tflite_with_select_tf_ops=true \
  --verbose_failures \
  --jobs=4 \
  //tensorflow/lite:libtensorflowlite.so

echo "Building libtensorflowlite_flex.so for ARM64..."
bazel build --config=elinux_aarch64 \
  -c opt \
  --config=monolithic \
  --define=tflite_with_select_tf_ops=true \
  --verbose_failures \
  --jobs=4 \
  //tensorflow/lite/delegates/flex:libtensorflowlite_flex.so

echo '=== 6. Verify ARM64 Binaries ==='
echo "Checking libtensorflowlite.so:"
file bazel-bin/tensorflow/lite/libtensorflowlite.so
if file bazel-bin/tensorflow/lite/libtensorflowlite.so | grep -q "aarch64\|ARM aarch64"; then
    echo "✓ Confirmed: ARM64 binary"
else
    echo "❌ ERROR: Not an ARM64 binary!"
    file bazel-bin/tensorflow/lite/libtensorflowlite.so
    exit 1
fi

echo "Checking libtensorflowlite_flex.so:"
file bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex.so
if file bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex.so | grep -q "aarch64\|ARM aarch64"; then
    echo "✓ Confirmed: ARM64 binary"
else
    echo "❌ ERROR: Not an ARM64 binary!"
    file bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex.so
    exit 1
fi

echo '=== 7. Packaging SDK ==='
mkdir -p /tmp/sdk/include
mkdir -p /tmp/sdk/lib

echo 'Copying Libraries...'
cp bazel-bin/tensorflow/lite/libtensorflowlite.so /tmp/sdk/lib/
cp bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex.so /tmp/sdk/lib/

echo 'Copying Source Headers...'
find tensorflow/lite \( -name '*.h' -o -name '*.hpp' -o -name '*.inc' \) -exec cp --parents {} /tmp/sdk/include/ \;

echo 'Copying Generated Headers...'
cd bazel-bin
find tensorflow/lite \( -name '*.pb.h' -o -name '*.inc' \) -exec cp --parents {} /tmp/sdk/include/ \;
cd /tensorflow_src

echo 'Copying External Dependencies...'

# FlatBuffers
FB_HEADER=$(find -L . -name 'flatbuffers.h' | grep 'external/flatbuffers' | head -n 1)
if [ -n "$FB_HEADER" ]; then
   FB_ROOT=$(dirname $(dirname "$FB_HEADER"))
   echo "Found FlatBuffers at: $FB_ROOT"
   cp -r "$FB_ROOT/flatbuffers" /tmp/sdk/include/
fi

# Abseil
ABSL_HEADER=$(find -L . -name 'string_view.h' | grep 'external/com_google_absl' | head -n 1)
if [ -n "$ABSL_HEADER" ]; then
   ABSL_ROOT=$(dirname $(dirname $(dirname "$ABSL_HEADER")))
   echo "Found Abseil at: $ABSL_ROOT"
   cp -r "$ABSL_ROOT/absl" /tmp/sdk/include/
fi

# Eigen
EIGEN_HEADER=$(find -L . -name 'Core' | grep 'external/eigen_archive/Eigen' | head -n 1)
if [ -n "$EIGEN_HEADER" ]; then
   EIGEN_ROOT=$(dirname $(dirname "$EIGEN_HEADER"))
   echo "Found Eigen at: $EIGEN_ROOT"
   cp -r "$EIGEN_ROOT/Eigen" /tmp/sdk/include/
   cp -r "$EIGEN_ROOT/unsupported" /tmp/sdk/include/ || true
fi

# Gemmlowp
GEMM_HEADER=$(find -L . -name 'fixedpoint.h' | grep 'external/gemmlowp' | head -n 1)
if [ -n "$GEMM_HEADER" ]; then
   GEMM_ROOT=$(dirname $(dirname "$GEMM_HEADER"))
   echo "Found Gemmlowp at: $GEMM_ROOT"
   mkdir -p /tmp/sdk/include/fixedpoint
   cp -r "$GEMM_ROOT/fixedpoint" /tmp/sdk/include/
   cp -r "$GEMM_ROOT/internal" /tmp/sdk/include/ || true
fi

# NEON_2_SSE
NEON_HEADER=$(find -L . -name 'NEON_2_SSE.h' | grep 'external' | head -n 1)
if [ -n "$NEON_HEADER" ]; then
   echo "Found NEON_2_SSE"
   cp "$NEON_HEADER" /tmp/sdk/include/
fi

echo '=== 8. Build Info ==='
cat > /tmp/sdk/BUILD_INFO.txt << 'INFO_EOF'
TensorFlow Lite SDK (ARM64)
===========================
Target: ARM64 (aarch64)
TensorFlow: 2.19.0
Bazel: 6.5.0
Build Config: --config=elinux_aarch64
Host: x86_64 Ubuntu 24.04
Cross-Compiled: Yes (using TensorFlow built-in toolchain)

Contents:
- lib/libtensorflowlite.so (ARM64)
- lib/libtensorflowlite_flex.so (ARM64)
- include/ (headers)

Requirements:
- Target system needs glibc 2.28 or higher

Usage on Raspberry Pi 4/5 or other ARM64 boards:
 g++ main.cpp \
  -I./include \
  -L./lib \
  -ltensorflowlite \
  -ltensorflowlite_flex \
  -lpthread -ldl -lm \
  -std=c++17 \
  -Wl,-rpath,'$ORIGIN/lib' \
  -o app

Verify binaries:
 file lib/libtensorflowlite.so
 # Should show: ARM aarch64, dynamically linked
INFO_EOF

echo '=== 9. Zip SDK ==='
cd /tmp
zip -r tflite_sdk_arm64.zip sdk/
ls -lh tflite_sdk_arm64.zip

echo '=== ✓ Build Complete ==='
echo "ARM64 SDK successfully built: /tmp/tflite_sdk_arm64.zip"
echo "Ready for deployment to ARM64 devices (Raspberry Pi 4/5, Coral, etc.)"
#!/bin/bash
set -e

echo '=== 1. Install Basic Tools & Cross-Compilation Tools ==='
apt-get update
apt-get install -y git wget curl build-essential python3 python3-pip python3-dev openjdk-11-jdk zip unzip \
  gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

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

echo '=== 5. Configure ARM64 CROSSTOOL ==='
mkdir -p tools/arm_compiler
cat > tools/arm_compiler/BUILD << 'BEOF'
package(default_visibility = ['//visibility:public'])
BEOF

cat > tools/arm_compiler/CROSSTOOL << 'CEOF'
major_version: "local"
minor_version: ""
default_target_cpu: "aarch64"

default_toolchain {
  cpu: "aarch64"
  toolchain_identifier: "aarch64-cross-toolchain"
}

toolchain {
  abi_version: "aarch64"
  abi_libc_version: "aarch64"
  builtin_sysroot: ""
  compiler: "compiler"
  host_system_name: "local"
  needsPic: true
  supports_gold_linker: false
  supports_incremental_linker: false
  supports_fission: false
  supports_interface_shared_objects: false
  supports_normalizing_ar: false
  supports_start_end_lib: false
  target_libc: "aarch64"
  target_cpu: "aarch64"
  target_system_name: "aarch64"
  toolchain_identifier: "aarch64-cross-toolchain"

  tool_path { name: "ar" path: "/usr/bin/aarch64-linux-gnu-ar" }
  tool_path { name: "compat-ld" path: "/usr/bin/aarch64-linux-gnu-ld" }
  tool_path { name: "cpp" path: "/usr/bin/aarch64-linux-gnu-cpp" }
  tool_path { name: "dwp" path: "/usr/bin/aarch64-linux-gnu-dwp" }
  tool_path { name: "gcc" path: "/usr/bin/aarch64-linux-gnu-gcc" }
  tool_path { name: "gcov" path: "/usr/bin/aarch64-linux-gnu-gcov" }
  tool_path { name: "ld" path: "/usr/bin/aarch64-linux-gnu-ld" }
  tool_path { name: "nm" path: "/usr/bin/aarch64-linux-gnu-nm" }
  tool_path { name: "objcopy" path: "/usr/bin/aarch64-linux-gnu-objcopy" }
  tool_path { name: "objdump" path: "/usr/bin/aarch64-linux-gnu-objdump" }
  tool_path { name: "strip" path: "/usr/bin/aarch64-linux-gnu-strip" }

  compiler_flag: "-U_FORTIFY_SOURCE"
  compiler_flag: "-D_FORTIFY_SOURCE=1"
  compiler_flag: "-fstack-protector"
  compiler_flag: "-Wall"
  compiler_flag: "-Wunused-but-set-parameter"
  compiler_flag: "-Wno-free-nonheap-object"
  compiler_flag: "-fno-omit-frame-pointer"
  compiler_flag: "-Wno-deprecated-declarations"

  cxx_flag: "-std=c++17"

  linker_flag: "-lstdc++"
  linker_flag: "-lm"
  linker_flag: "-lpthread"

  cxx_builtin_include_directory: "/usr/aarch64-linux-gnu/include"
  cxx_builtin_include_directory: "/usr/lib/gcc-cross/aarch64-linux-gnu/13/include"

  objcopy_embed_flag: "-I"
  objcopy_embed_flag: "binary"

  unfiltered_cxx_flag: "-fno-canonical-system-headers"
  unfiltered_cxx_flag: "-Wno-builtin-macro-redefined"
  unfiltered_cxx_flag: "-D__DATE__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIMESTAMP__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIME__=\"redacted\""

  compilation_mode_flags {
    mode: DBG
    compiler_flag: "-g"
  }
  compilation_mode_flags {
    mode: OPT
    compiler_flag: "-g0"
    compiler_flag: "-O3"
    compiler_flag: "-DNDEBUG"
    compiler_flag: "-ffunction-sections"
    compiler_flag: "-fdata-sections"
    linker_flag: "-Wl,--gc-sections"
  }

  linking_mode_flags { mode: DYNAMIC }
}
CEOF

echo '✓ CROSSTOOL configured'

echo '=== 6. Configure .bazelrc ==='
cat >> .bazelrc << 'BZEF'

# ARM64 Cross-Compilation Configuration
build:rpi --crosstool_top=//tools/arm_compiler:toolchain
build:rpi --cpu=aarch64
build:rpi --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
build:rpi --copt=-march=armv8-a
build:rpi --copt=-O3
BZEF

echo '✓ .bazelrc configured'

echo '=== 7. Build for ARM64 ==='
bazel build -c opt \
  --config=monolithic \
  --config=rpi \
  --define=tflite_with_select_tf_ops=true \
  --verbose_failures \
  --jobs=4 \
  //tensorflow/lite:libtensorflowlite.so \
  //tensorflow/lite/delegates/flex:libtensorflowlite_flex.so

echo '=== 8. Verify ARM64 Binaries ==='
file bazel-bin/tensorflow/lite/libtensorflowlite.so
file bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex.so

echo '=== 9. Packaging SDK ==='
mkdir -p /tmp/sdk/include
mkdir -p /tmp/sdk/lib

echo 'Copying Libraries...'
cp bazel-bin/tensorflow/lite/libtensorflowlite.so /tmp/sdk/lib/
cp bazel-bin/tensorflow/lite/delegates/flex/libtensorflowlite_flex.so /tmp/sdk/lib/

echo 'Copying Source Headers...'
find tensorflow \( -name '*.h' -o -name '*.hpp' -o -name '*.inc' \) -exec cp --parents {} /tmp/sdk/include/ \;

echo 'Copying Generated Headers...'
cd bazel-bin
find tensorflow \( -name '*.pb.h' -o -name '*.inc' \) -exec cp --parents {} /tmp/sdk/include/ \;
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

echo '=== 10. Build Info ==='
cat > /tmp/sdk/BUILD_INFO.txt << 'INFO_EOF'
TensorFlow Lite SDK (ARM64)
===========================
Target: ARM64 (aarch64)
TensorFlow: 2.19.0
Bazel: 6.5.0
Host: x86_64 Ubuntu 24.04
Cross-Compiled: Yes

Contents:
- lib/libtensorflowlite.so (ARM64)
- lib/libtensorflowlite_flex.so (ARM64)
- include/ (headers)

Usage on Raspberry Pi 4/5:
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
 # Should show: ARM aarch64
INFO_EOF

echo '=== 11. Zip SDK ==='
cd /tmp
zip -r tflite_sdk_arm64.zip sdk/
ls -lh tflite_sdk_arm64.zip
# 라즈베리파이 cpp 테스트트

## 요구사항
- Raspberry Pi 4/5 (64-bit OS)
- Raspberry Pi OS Bullseye 이상
- glibc 2.28+

## 1. 스크립트 사용 방법

### 1.1 라즈베리파이에서 git clone - 파일 다운로드
```bash
git clone https://github.com/solmouse/IncrementalLearning.git
cd IncrementalLearning/pi/cpp_pi_test_251212/
```

### 1.2 디렉토리 구조 확인
```
IncrementalLearning/
└── pi/
    └── cpp_pi_test_251212/
        ├── tflite_sdk_arm64.tar.gz
        └── setup_and_run.sh
```

### 1.3 실행
```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

## 2. 스크립트 사용 안 될 경우
```bash
# 우선 1번과 동일한 경로 진입
cd IncrementalLearning/pi/cpp_pi_test_251212/

# 압축 해제
tar -xzvf tflite_sdk_arm64.tar.gz
cd sdk

# 컴파일 (변경)
g++ tests/phase2_with_flex.cpp tests/cnpy.cpp \
    -I./include -L./lib \
    -ltensorflowlite \
    -lpthread -ldl -lm -lz \
    -std=c++17 -O3 \
    -Wl,-rpath,'$ORIGIN/../lib' \
    -o tests/phase2_pi

# 실행 (변경)
cd tests
export LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH

./phase2_pi \
  ../models/model.tflite \
  ../models/ckpt_before.npy \
  ../models/ckpt_after.npy \
  ../data/domainB_images.npy \
  ../data/domainB_labels.npy \
  ../data/domainA_images.npy \
  ../data/domainA_labels.npy
```

## 3. 문제 해결

### 3.1 glibc 버전 확인
```bash
ldd --version # 2.28 이상이어야 함
```

### 3.2 라이브러리 경로 오류
```bash
export LD_LIBRARY_PATH=$PWD/sdk/lib:$LD_LIBRARY_PATH
```
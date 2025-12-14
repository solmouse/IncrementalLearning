# 251214 변경사항

1. **dlopen 제거**, 컴파일 시 링크
    - dlopen 사용(동적로딩) 시 문제
        - **안정성 문제**(프로그램 실행되던 중 런타임에 에러 생길 수 있음)
        - 라이브러리 경로 하드코딩
    - 링크 방식 
        - 처음부터 라이브러리 없으면 바로 에러, 실행되지 않아서 문제 조기 발견 가능
2. **명시적 연산자 등록 제거**
    - tflite::ops::builtin::BuiltinOpResolver resolver(295줄)을 통해 TFlite가 지원하는 모든 Native 연산자 등록
    - 작동 방식
        - BuiltinOpResolver가 아는 연산자(Conv, Add 등 기본 수학 연산) 가져감
        - 모르는 연산자(FlexBroadcastGradientArgs 미분 계산 연산자와 같은 고급 연산자)는 Flex가 처리


# 컴파일 방식
```bash
g++ -o phase2_pi     phase2_with_flex.cpp     cnpy.cpp     -I.     -I../include     -L../lib     -Wl,--no-as-needed -ltensorflowlite_flex -Wl,--as-needed     -ltensorflowlite     -lpthread     -lm     -lz     -std=c++17     -Wl,-rpath,'$ORIGIN/../lib'
```

# 실행 방식
```bash
timeout 300 ./phase2_pi     ../models/model.tflite     ../models/ckpt_before.npy     ../models/ckpt_after.npy     ../../data/domainB_images.npy     ../../data/domainB_labels.npy     ../../data/domainA_images.npy     ../../data/domainA_labels.npy
```
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <random>
#include <algorithm>
#include <map>
#include <numeric>
#include <cmath>
#include <dlfcn.h>
#include "cnpy.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/signature_runner.h"

// ============================================================================
// TfLite 연산자 Register
// ============================================================================
namespace tflite {
namespace ops {
namespace builtin {
  // --- 수학 및 기본 연산 ---
  TfLiteRegistration* Register_ADD();
  TfLiteRegistration* Register_SUB();
  TfLiteRegistration* Register_MUL();
  TfLiteRegistration* Register_DIV();
  TfLiteRegistration* Register_POW();
  TfLiteRegistration* Register_SQUARE();
  TfLiteRegistration* Register_SQRT();
  TfLiteRegistration* Register_RSQRT();
  TfLiteRegistration* Register_EXP();
  TfLiteRegistration* Register_LOG();
  TfLiteRegistration* Register_ABS();
  TfLiteRegistration* Register_NEG();
  TfLiteRegistration* Register_MINIMUM();
  TfLiteRegistration* Register_MAXIMUM();

  // --- 신경망 레이어 ---
  TfLiteRegistration* Register_CONV_2D();
  TfLiteRegistration* Register_DEPTHWISE_CONV_2D();
  TfLiteRegistration* Register_FULLY_CONNECTED();
  TfLiteRegistration* Register_SOFTMAX();
  TfLiteRegistration* Register_LOGISTIC();
  TfLiteRegistration* Register_TANH();
  TfLiteRegistration* Register_RELU();
  TfLiteRegistration* Register_RELU6();
  TfLiteRegistration* Register_MAX_POOL_2D();
  TfLiteRegistration* Register_AVERAGE_POOL_2D();
  TfLiteRegistration* Register_L2_NORMALIZATION();
  TfLiteRegistration* Register_LOCAL_RESPONSE_NORMALIZATION();

  // --- 텐서 조작 및 형상 변경 ---
  TfLiteRegistration* Register_RESHAPE();
  TfLiteRegistration* Register_TRANSPOSE();
  TfLiteRegistration* Register_SHAPE();
  TfLiteRegistration* Register_CAST();
  TfLiteRegistration* Register_SLICE();
  TfLiteRegistration* Register_STRIDED_SLICE();
  TfLiteRegistration* Register_CONCATENATION();
  TfLiteRegistration* Register_PACK();
  TfLiteRegistration* Register_UNPACK();
  TfLiteRegistration* Register_SPLIT();
  TfLiteRegistration* Register_SQUEEZE();
  TfLiteRegistration* Register_EXPAND_DIMS();
  TfLiteRegistration* Register_GATHER();
  TfLiteRegistration* Register_SELECT();
  TfLiteRegistration* Register_TILE();
  TfLiteRegistration* Register_PAD();
  TfLiteRegistration* Register_FILL();
  TfLiteRegistration* Register_ZEROS_LIKE();
  TfLiteRegistration* Register_ONE_HOT();
  TfLiteRegistration* Register_ARG_MAX();
  TfLiteRegistration* Register_ARG_MIN();
  TfLiteRegistration* Register_RESIZE_BILINEAR();
  TfLiteRegistration* Register_RESIZE_NEAREST_NEIGHBOR();

  // --- 비교 및 논리 ---
  TfLiteRegistration* Register_LESS();
  TfLiteRegistration* Register_LESS_EQUAL();
  TfLiteRegistration* Register_GREATER();
  TfLiteRegistration* Register_GREATER_EQUAL();
  TfLiteRegistration* Register_EQUAL();
  TfLiteRegistration* Register_NOT_EQUAL();
  TfLiteRegistration* Register_LOGICAL_AND();
  TfLiteRegistration* Register_LOGICAL_OR();
  TfLiteRegistration* Register_LOGICAL_NOT();

  // --- 리덕션 ---
  TfLiteRegistration* Register_MEAN();
  TfLiteRegistration* Register_SUM();
  TfLiteRegistration* Register_REDUCE_MAX();
  TfLiteRegistration* Register_REDUCE_MIN();
  TfLiteRegistration* Register_REDUCE_ANY();
  TfLiteRegistration* Register_REDUCE_PROD();
  TfLiteRegistration* Register_REDUCE_ALL();
  
  // --- 변수 ---
  TfLiteRegistration* Register_VAR_HANDLE();
  TfLiteRegistration* Register_READ_VARIABLE();
  TfLiteRegistration* Register_ASSIGN_VARIABLE();

  TfLiteRegistration* Register_BATCH_MATMUL();
  TfLiteRegistration* Register_LOG_SOFTMAX();
  TfLiteRegistration* Register_SELECT_V2();
  TfLiteRegistration* Register_BROADCAST_TO();
  TfLiteRegistration* Register_BROADCAST_ARGS();

  // --- 수학 및 손실 함수 관련 ---
  TfLiteRegistration* Register_SQUARED_DIFFERENCE();
  TfLiteRegistration* Register_CEIL();
  TfLiteRegistration* Register_FLOOR();
  TfLiteRegistration* Register_ROUND();
  TfLiteRegistration* Register_COS();
  TfLiteRegistration* Register_SIN();

  // --- 텐서 조작 고급 ---
  TfLiteRegistration* Register_SPLIT_V();
  TfLiteRegistration* Register_REVERSE_V2();
  TfLiteRegistration* Register_RANK();
  TfLiteRegistration* Register_RANGE();
  TfLiteRegistration* Register_MIRROR_PAD();
  TfLiteRegistration* Register_TOPK_V2();
  TfLiteRegistration* Register_UNIQUE();
  TfLiteRegistration* Register_GATHER_ND();
  TfLiteRegistration* Register_SCATTER_ND();
  TfLiteRegistration* Register_WHERE();

  // -- Base 열어줄 때 추가 ---
  TfLiteRegistration* Register_TRANSPOSE_CONV();
  TfLiteRegistration* Register_PADV2();
  TfLiteRegistration* Register_ADD_N();
}
}
}

// ============================================================================
// 상수 정의
// ============================================================================
const int IMG_SIZE = 224;
const int NUM_CLASSES = 4;
const int BATCH_SIZE = 32;
const int INCR_NEW_STEPS = 60;
const int INCR_REHEARSAL_STEPS = 24;
const int EXEMPLARS_PER_CLASS = 16;

// ============================================================================
// NPY 파일 유틸리티 (cnpy 사용)
// ============================================================================
std::vector<float> loadNpy(const char* filename) {
    try {
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        
        if (arr.word_size != sizeof(float)) {
            std::cerr << "Error: NPY file is not float32 type (word_size=" 
                      << arr.word_size << ")" << std::endl;
            return {};
        }
        
        float* data_ptr = arr.data<float>();
        size_t num_floats = arr.num_vals;
        
        std::vector<float> data(data_ptr, data_ptr + num_floats);
        
        std::cout << "Loaded " << num_floats << " floats from " << filename << std::endl;
        
        if (num_floats > 0) {
            float min_val = *std::min_element(data.begin(), data.end());
            float max_val = *std::max_element(data.begin(), data.end());
            
            if (std::isnan(min_val) || std::isinf(min_val) || std::abs(max_val) > 1e6) {
                std::cerr << "⚠ WARNING: Data looks corrupted" << std::endl;
                std::cerr << "  Range: [" << min_val << ", " << max_val << "]" << std::endl;
                return {};
            }
        }
        
        return data;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading NPY file: " << e.what() << std::endl;
        return {};
    }
}

bool saveNpy(const char* filename, const std::vector<float>& data) {
    try {
        std::vector<size_t> shape = {data.size()};
        cnpy::npy_save(std::string(filename), data.data(), shape, "w");
        std::cout << "Saved " << data.size() << " floats to " << filename << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving NPY file: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Exemplar Manager (리허설용)
// ============================================================================
struct Exemplar {
    std::vector<float> image;
    std::vector<float> label;
    int class_id;
};

class ExemplarManager {
public:
    ExemplarManager(int capacity_per_class = EXEMPLARS_PER_CLASS) 
        : capacity_(capacity_per_class) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            bank_[i] = std::vector<Exemplar>();
        }
    }
    
    void addCandidates(const std::vector<float>& images, const std::vector<float>& labels, int batch_size) {
        if (capacity_ <= 0) return;
        
        int img_size = IMG_SIZE * IMG_SIZE * 3;
        
        for (int i = 0; i < batch_size; i++) {
            std::vector<float> img(images.begin() + i * img_size, 
                                  images.begin() + (i + 1) * img_size);
            
            std::vector<float> lbl(labels.begin() + i * NUM_CLASSES,
                                  labels.begin() + (i + 1) * NUM_CLASSES);
            
            int class_id = 0;
            float max_val = lbl[0];
            for (int c = 1; c < NUM_CLASSES; c++) {
                if (lbl[c] > max_val) {
                    max_val = lbl[c];
                    class_id = c;
                }
            }
            
            Exemplar ex;
            ex.image = img;
            ex.label = lbl;
            ex.class_id = class_id;
            
            bank_[class_id].push_back(ex);
        }
        
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (bank_[c].size() > capacity_) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::shuffle(bank_[c].begin(), bank_[c].end(), gen);
                bank_[c].resize(capacity_);
            }
        }
    }
    
    bool getRehearsalBatch(std::vector<float>& images, std::vector<float>& labels, int batch_size) {
        std::vector<Exemplar> all_exemplars;
        for (int c = 0; c < NUM_CLASSES; c++) {
            all_exemplars.insert(all_exemplars.end(), bank_[c].begin(), bank_[c].end());
        }
        
        if (all_exemplars.empty()) return false;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(all_exemplars.begin(), all_exemplars.end(), gen);
        
        int actual_size = std::min(batch_size, (int)all_exemplars.size());
        
        images.clear();
        labels.clear();
        
        for (int i = 0; i < actual_size; i++) {
            images.insert(images.end(), all_exemplars[i].image.begin(), all_exemplars[i].image.end());
            labels.insert(labels.end(), all_exemplars[i].label.begin(), all_exemplars[i].label.end());
        }
        
        return true;
    }
    
    int getTotalSize() const {
        int total = 0;
        for (const auto& pair : bank_) {
            total += pair.second.size();
        }
        return total;
    }

private:
    int capacity_;
    std::map<int, std::vector<Exemplar>> bank_;
};

// ============================================================================
// 더미 데이터 생성 (fallback용)
// ============================================================================
void generateDummyBatch(std::vector<float>& images, std::vector<float>& labels, int batch_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.5f, 0.15f);
    
    images.resize(batch_size * IMG_SIZE * IMG_SIZE * 3);
    for (size_t i = 0; i < images.size(); i++) {
        float val = dist(gen);
        images[i] = std::max(0.0f, std::min(1.0f, val));
    }
    
    labels.resize(batch_size * NUM_CLASSES);
    std::uniform_int_distribution<int> class_dist(0, NUM_CLASSES - 1);
    
    for (int i = 0; i < batch_size; i++) {
        int target_class = class_dist(gen);
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (c == target_class) {
                labels[i * NUM_CLASSES + c] = 0.9f;
            } else {
                labels[i * NUM_CLASSES + c] = 0.1f / (NUM_CLASSES - 1);
            }
        }
    }
}

// ============================================================================
// 평가 함수 (정확도 계산)
// ============================================================================
float evaluateAccuracy(tflite::SignatureRunner* infer_runner, 
                      const std::vector<float>& images, 
                      const std::vector<float>& labels,
                      int num_samples) {
    
    int img_size = IMG_SIZE * IMG_SIZE * 3;
    int correct = 0;
    int total = 0;
    
    // 배치 단위로 평가
    for (int start = 0; start < num_samples; start += BATCH_SIZE) {
        int batch_size = std::min(BATCH_SIZE, num_samples - start);
        int end = std::min(start + batch_size, num_samples);
        
        // 입력 데이터 준비
        std::vector<float> batch_images(
            images.begin() + start * img_size,
            images.begin() + end * img_size
        );
        
        std::vector<float> batch_labels(
            labels.begin() + start * NUM_CLASSES,
            labels.begin() + end * NUM_CLASSES
        );
        
        // Resize if needed
        if (batch_size != BATCH_SIZE) {
            std::vector<int> x_shape = {batch_size, IMG_SIZE, IMG_SIZE, 3};
            infer_runner->ResizeInputTensor("x", x_shape);
            infer_runner->AllocateTensors();
        }
        
        // 추론
        auto* x_tensor = infer_runner->input_tensor("x");
        std::memcpy(x_tensor->data.f, batch_images.data(), batch_images.size() * sizeof(float));
        infer_runner->Invoke();
        
        // 결과 확인
        auto* output_tensor = infer_runner->output_tensor("output");
        float* probs = output_tensor->data.f;
        
        for (int i = 0; i < batch_size; i++) {
            // Predicted class
            int pred_class = 0;
            float max_prob = probs[i * NUM_CLASSES];
            for (int c = 1; c < NUM_CLASSES; c++) {
                if (probs[i * NUM_CLASSES + c] > max_prob) {
                    max_prob = probs[i * NUM_CLASSES + c];
                    pred_class = c;
                }
            }
            
            // True class
            int true_class = 0;
            float max_label = batch_labels[i * NUM_CLASSES];
            for (int c = 1; c < NUM_CLASSES; c++) {
                if (batch_labels[i * NUM_CLASSES + c] > max_label) {
                    max_label = batch_labels[i * NUM_CLASSES + c];
                    true_class = c;
                }
            }
            
            if (pred_class == true_class) correct++;
            total++;
        }
        
        // Restore batch size
        if (batch_size != BATCH_SIZE) {
            std::vector<int> x_shape = {BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3};
            infer_runner->ResizeInputTensor("x", x_shape);
            infer_runner->AllocateTensors();
        }
    }
    
    return (total > 0) ? (float)correct / total : 0.0f;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "[Phase-2] TFLite 증분학습 (with Flex)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <model.tflite> <ckpt_before.npy> [ckpt_after.npy] "
                  << "[domainB_images.npy] [domainB_labels.npy] "
                  << "[domainA_images.npy] [domainA_labels.npy]" 
                  << std::endl;
        return -1;
    }
    
    const char* model_path = argv[1];
    const char* ckpt_before = argv[2];
    const char* ckpt_after = (argc >= 4) ? argv[3] : "../models/ckpt_after.npy";
    const char* domainB_images_path = (argc >= 5) ? argv[4] : "../../data/domainB_images.npy";
    const char* domainB_labels_path = (argc >= 6) ? argv[5] : "../../data/domainB_labels.npy";
    const char* domainA_images_path = (argc >= 7) ? argv[6] : "../../data/domainA_images.npy";
    const char* domainA_labels_path = (argc >= 8) ? argv[7] : "../../data/domainA_labels.npy";
    
    std::cout << "\nModel: " << model_path << std::endl;
    std::cout << "Checkpoint (before): " << ckpt_before << std::endl;
    std::cout << "Checkpoint (after): " << ckpt_after << std::endl;
    std::cout << "DomainB images: " << domainB_images_path << std::endl;
    std::cout << "DomainB labels: " << domainB_labels_path << std::endl;
    std::cout << "DomainA images: " << domainA_images_path << std::endl;
    std::cout << "DomainA labels: " << domainA_labels_path << std::endl;
    
    // ========================================
    // Flex delegate 로드
    // ========================================
    std::cout << "\n--- Pre-loading Flex Delegate ---" << std::endl;
    void* flex_lib = dlopen("libtensorflowlite_flex.so", RTLD_NOW | RTLD_GLOBAL);
    if (!flex_lib) {
        std::cerr << "Failed to load Flex delegate: " << dlerror() << std::endl;
        return -1;
    }
    std::cout << "✓ Flex delegate preloaded" << std::endl;
    
    // ========================================
    // 모델 로드
    // ========================================
    std::cout << "\n--- Loading Model ---" << std::endl;
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }
    std::cout << "✓ Model loaded" << std::endl;
    
    tflite::MutableOpResolver resolver;

    resolver.AddBuiltin(tflite::BuiltinOperator_ADD, tflite::ops::builtin::Register_ADD());
    resolver.AddBuiltin(tflite::BuiltinOperator_SUB, tflite::ops::builtin::Register_SUB());
    resolver.AddBuiltin(tflite::BuiltinOperator_MUL, tflite::ops::builtin::Register_MUL());
    resolver.AddBuiltin(tflite::BuiltinOperator_DIV, tflite::ops::builtin::Register_DIV());
    resolver.AddBuiltin(tflite::BuiltinOperator_POW, tflite::ops::builtin::Register_POW());
    resolver.AddBuiltin(tflite::BuiltinOperator_SQUARE, tflite::ops::builtin::Register_SQUARE());
    resolver.AddBuiltin(tflite::BuiltinOperator_SQRT, tflite::ops::builtin::Register_SQRT());
    resolver.AddBuiltin(tflite::BuiltinOperator_RSQRT, tflite::ops::builtin::Register_RSQRT());
    resolver.AddBuiltin(tflite::BuiltinOperator_EXP, tflite::ops::builtin::Register_EXP());
    resolver.AddBuiltin(tflite::BuiltinOperator_LOG, tflite::ops::builtin::Register_LOG());
    resolver.AddBuiltin(tflite::BuiltinOperator_ABS, tflite::ops::builtin::Register_ABS());
    resolver.AddBuiltin(tflite::BuiltinOperator_NEG, tflite::ops::builtin::Register_NEG());
    resolver.AddBuiltin(tflite::BuiltinOperator_MINIMUM, tflite::ops::builtin::Register_MINIMUM());
    resolver.AddBuiltin(tflite::BuiltinOperator_MAXIMUM, tflite::ops::builtin::Register_MAXIMUM());

    resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D, tflite::ops::builtin::Register_CONV_2D());
    resolver.AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D, tflite::ops::builtin::Register_DEPTHWISE_CONV_2D());
    resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED, tflite::ops::builtin::Register_FULLY_CONNECTED());
    resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX, tflite::ops::builtin::Register_SOFTMAX());
    resolver.AddBuiltin(tflite::BuiltinOperator_LOGISTIC, tflite::ops::builtin::Register_LOGISTIC());
    resolver.AddBuiltin(tflite::BuiltinOperator_TANH, tflite::ops::builtin::Register_TANH());
    resolver.AddBuiltin(tflite::BuiltinOperator_RELU, tflite::ops::builtin::Register_RELU());
    resolver.AddBuiltin(tflite::BuiltinOperator_RELU6, tflite::ops::builtin::Register_RELU6());
    resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D, tflite::ops::builtin::Register_MAX_POOL_2D());
    resolver.AddBuiltin(tflite::BuiltinOperator_AVERAGE_POOL_2D, tflite::ops::builtin::Register_AVERAGE_POOL_2D());
    resolver.AddBuiltin(tflite::BuiltinOperator_L2_NORMALIZATION, tflite::ops::builtin::Register_L2_NORMALIZATION());
    resolver.AddBuiltin(tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION, tflite::ops::builtin::Register_LOCAL_RESPONSE_NORMALIZATION());

    resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE, tflite::ops::builtin::Register_RESHAPE());
    resolver.AddBuiltin(tflite::BuiltinOperator_TRANSPOSE, tflite::ops::builtin::Register_TRANSPOSE());
    resolver.AddBuiltin(tflite::BuiltinOperator_SHAPE, tflite::ops::builtin::Register_SHAPE());
    resolver.AddBuiltin(tflite::BuiltinOperator_CAST, tflite::ops::builtin::Register_CAST());
    resolver.AddBuiltin(tflite::BuiltinOperator_SLICE, tflite::ops::builtin::Register_SLICE());
    resolver.AddBuiltin(tflite::BuiltinOperator_STRIDED_SLICE, tflite::ops::builtin::Register_STRIDED_SLICE());
    resolver.AddBuiltin(tflite::BuiltinOperator_CONCATENATION, tflite::ops::builtin::Register_CONCATENATION());
    resolver.AddBuiltin(tflite::BuiltinOperator_PACK, tflite::ops::builtin::Register_PACK());
    resolver.AddBuiltin(tflite::BuiltinOperator_UNPACK, tflite::ops::builtin::Register_UNPACK());
    resolver.AddBuiltin(tflite::BuiltinOperator_SPLIT, tflite::ops::builtin::Register_SPLIT());
    resolver.AddBuiltin(tflite::BuiltinOperator_SQUEEZE, tflite::ops::builtin::Register_SQUEEZE());
    resolver.AddBuiltin(tflite::BuiltinOperator_EXPAND_DIMS, tflite::ops::builtin::Register_EXPAND_DIMS());
    resolver.AddBuiltin(tflite::BuiltinOperator_GATHER, tflite::ops::builtin::Register_GATHER());
    resolver.AddBuiltin(tflite::BuiltinOperator_SELECT, tflite::ops::builtin::Register_SELECT());
    resolver.AddBuiltin(tflite::BuiltinOperator_TILE, tflite::ops::builtin::Register_TILE());
    resolver.AddBuiltin(tflite::BuiltinOperator_PAD, tflite::ops::builtin::Register_PAD());
    resolver.AddBuiltin(tflite::BuiltinOperator_FILL, tflite::ops::builtin::Register_FILL());
    resolver.AddBuiltin(tflite::BuiltinOperator_ZEROS_LIKE, tflite::ops::builtin::Register_ZEROS_LIKE());
    resolver.AddBuiltin(tflite::BuiltinOperator_ONE_HOT, tflite::ops::builtin::Register_ONE_HOT());
    resolver.AddBuiltin(tflite::BuiltinOperator_ARG_MAX, tflite::ops::builtin::Register_ARG_MAX());
    resolver.AddBuiltin(tflite::BuiltinOperator_ARG_MIN, tflite::ops::builtin::Register_ARG_MIN());
    resolver.AddBuiltin(tflite::BuiltinOperator_RESIZE_BILINEAR, tflite::ops::builtin::Register_RESIZE_BILINEAR());
    resolver.AddBuiltin(tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR, tflite::ops::builtin::Register_RESIZE_NEAREST_NEIGHBOR());

    resolver.AddBuiltin(tflite::BuiltinOperator_LESS, tflite::ops::builtin::Register_LESS());
    resolver.AddBuiltin(tflite::BuiltinOperator_LESS_EQUAL, tflite::ops::builtin::Register_LESS_EQUAL());
    resolver.AddBuiltin(tflite::BuiltinOperator_GREATER, tflite::ops::builtin::Register_GREATER());
    resolver.AddBuiltin(tflite::BuiltinOperator_GREATER_EQUAL, tflite::ops::builtin::Register_GREATER_EQUAL());
    resolver.AddBuiltin(tflite::BuiltinOperator_EQUAL, tflite::ops::builtin::Register_EQUAL());
    resolver.AddBuiltin(tflite::BuiltinOperator_NOT_EQUAL, tflite::ops::builtin::Register_NOT_EQUAL());
    resolver.AddBuiltin(tflite::BuiltinOperator_LOGICAL_AND, tflite::ops::builtin::Register_LOGICAL_AND());
    resolver.AddBuiltin(tflite::BuiltinOperator_LOGICAL_OR, tflite::ops::builtin::Register_LOGICAL_OR());
    resolver.AddBuiltin(tflite::BuiltinOperator_LOGICAL_NOT, tflite::ops::builtin::Register_LOGICAL_NOT());

    resolver.AddBuiltin(tflite::BuiltinOperator_MEAN, tflite::ops::builtin::Register_MEAN());
    resolver.AddBuiltin(tflite::BuiltinOperator_SUM, tflite::ops::builtin::Register_SUM());
    resolver.AddBuiltin(tflite::BuiltinOperator_REDUCE_MAX, tflite::ops::builtin::Register_REDUCE_MAX());
    resolver.AddBuiltin(tflite::BuiltinOperator_REDUCE_MIN, tflite::ops::builtin::Register_REDUCE_MIN());
    resolver.AddBuiltin(tflite::BuiltinOperator_REDUCE_ANY, tflite::ops::builtin::Register_REDUCE_ANY());
    resolver.AddBuiltin(tflite::BuiltinOperator_REDUCE_PROD, tflite::ops::builtin::Register_REDUCE_PROD());
    resolver.AddBuiltin(tflite::BuiltinOperator_REDUCE_ALL, tflite::ops::builtin::Register_REDUCE_ALL());

    resolver.AddBuiltin(tflite::BuiltinOperator_VAR_HANDLE, tflite::ops::builtin::Register_VAR_HANDLE());
    resolver.AddBuiltin(tflite::BuiltinOperator_READ_VARIABLE, tflite::ops::builtin::Register_READ_VARIABLE());
    resolver.AddBuiltin(tflite::BuiltinOperator_ASSIGN_VARIABLE, tflite::ops::builtin::Register_ASSIGN_VARIABLE());

    resolver.AddBuiltin(tflite::BuiltinOperator_BATCH_MATMUL, tflite::ops::builtin::Register_BATCH_MATMUL());
    resolver.AddBuiltin(tflite::BuiltinOperator_LOG_SOFTMAX, tflite::ops::builtin::Register_LOG_SOFTMAX());
    resolver.AddBuiltin(tflite::BuiltinOperator_SELECT_V2, tflite::ops::builtin::Register_SELECT_V2());
    resolver.AddBuiltin(tflite::BuiltinOperator_BROADCAST_TO, tflite::ops::builtin::Register_BROADCAST_TO());
    resolver.AddBuiltin(tflite::BuiltinOperator_BROADCAST_ARGS, tflite::ops::builtin::Register_BROADCAST_ARGS());

    resolver.AddBuiltin(tflite::BuiltinOperator_SQUARED_DIFFERENCE, tflite::ops::builtin::Register_SQUARED_DIFFERENCE());
    resolver.AddBuiltin(tflite::BuiltinOperator_CEIL, tflite::ops::builtin::Register_CEIL());
    resolver.AddBuiltin(tflite::BuiltinOperator_FLOOR, tflite::ops::builtin::Register_FLOOR());
    resolver.AddBuiltin(tflite::BuiltinOperator_ROUND, tflite::ops::builtin::Register_ROUND());
    resolver.AddBuiltin(tflite::BuiltinOperator_COS, tflite::ops::builtin::Register_COS());
    resolver.AddBuiltin(tflite::BuiltinOperator_SIN, tflite::ops::builtin::Register_SIN());

    resolver.AddBuiltin(tflite::BuiltinOperator_SPLIT_V, tflite::ops::builtin::Register_SPLIT_V());
    resolver.AddBuiltin(tflite::BuiltinOperator_REVERSE_V2, tflite::ops::builtin::Register_REVERSE_V2());
    resolver.AddBuiltin(tflite::BuiltinOperator_RANK, tflite::ops::builtin::Register_RANK());
    resolver.AddBuiltin(tflite::BuiltinOperator_RANGE, tflite::ops::builtin::Register_RANGE());
    resolver.AddBuiltin(tflite::BuiltinOperator_MIRROR_PAD, tflite::ops::builtin::Register_MIRROR_PAD());
    resolver.AddBuiltin(tflite::BuiltinOperator_TOPK_V2, tflite::ops::builtin::Register_TOPK_V2());
    resolver.AddBuiltin(tflite::BuiltinOperator_UNIQUE, tflite::ops::builtin::Register_UNIQUE());
    resolver.AddBuiltin(tflite::BuiltinOperator_GATHER_ND, tflite::ops::builtin::Register_GATHER_ND());
    resolver.AddBuiltin(tflite::BuiltinOperator_SCATTER_ND, tflite::ops::builtin::Register_SCATTER_ND());
    resolver.AddBuiltin(tflite::BuiltinOperator_WHERE, tflite::ops::builtin::Register_WHERE());

    resolver.AddBuiltin(tflite::BuiltinOperator_TRANSPOSE_CONV, tflite::ops::builtin::Register_TRANSPOSE_CONV());
    resolver.AddBuiltin(tflite::BuiltinOperator_PADV2, tflite::ops::builtin::Register_PADV2());
    resolver.AddBuiltin(tflite::BuiltinOperator_ADD_N, tflite::ops::builtin::Register_ADD_N()
);

    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    if (!interpreter) {
        std::cerr << "Failed to create interpreter" << std::endl;
        return -1;
    }
    interpreter->SetNumThreads(4);
    std::cout << "✓ Interpreter created" << std::endl;
    
    // ========================================
    // Initialize
    // ========================================
    std::cout << "\n--- Initialize ---" << std::endl;
    auto* init_runner = interpreter->GetSignatureRunner("initialize");
    if (!init_runner) {
        std::cerr << "Failed to get initialize runner" << std::endl;
        return -1;
    }
    init_runner->AllocateTensors();
    init_runner->Invoke();
    std::cout << "✓ Model initialized" << std::endl;
    
    // ========================================
    // Restore Checkpoint
    // ========================================
    std::cout << "\n--- Restore Checkpoint ---" << std::endl;
    auto* restore_runner = interpreter->GetSignatureRunner("restore");
    if (!restore_runner) {
        std::cerr << "Failed to get restore runner" << std::endl;
        return -1;
    }

    auto weights = loadNpy(ckpt_before);
    if (weights.empty()) {
        std::cerr << "Failed to load checkpoint file." << std::endl;
        return -1;
    }
    
    std::vector<int> new_shape = { static_cast<int>(weights.size()) };
    
    if (restore_runner->ResizeInputTensor("flat_weights", new_shape) != kTfLiteOk) {
        std::cerr << "Failed to resize restore input tensor." << std::endl;
        return -1;
    }

    if (restore_runner->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors for restore runner." << std::endl;
        return -1;
    }
    
    auto* weights_tensor = restore_runner->input_tensor("flat_weights");
    if (weights_tensor->bytes != weights.size() * sizeof(float)) {
        std::cerr << "Size mismatch after resize!" << std::endl;
        return -1;
    }
    
    std::memcpy(weights_tensor->data.f, weights.data(), weights.size() * sizeof(float));
    
    if (restore_runner->Invoke() != kTfLiteOk) {
        std::cerr << "Error: Restore Invoke failed." << std::endl;
        return -1;
    }
    std::cout << "✓ Checkpoint restored" << std::endl;
    
    // ========================================
    // 실제 데이터 로드
    // ========================================
    std::cout << "\n--- Loading Real Data ---" << std::endl;
    
    // DomainB (새 도메인)
    auto domainB_images = loadNpy(domainB_images_path);
    auto domainB_labels = loadNpy(domainB_labels_path);
    
    if (domainB_images.empty() || domainB_labels.empty()) {
        std::cerr << "⚠ WARNING: Failed to load domainB data" << std::endl;
        return -1;
    }
    
    std::cout << "✓ DomainB loaded" << std::endl;
    std::cout << "  Images: " << domainB_images.size() << " floats" << std::endl;
    std::cout << "  Labels: " << domainB_labels.size() << " floats" << std::endl;
    
    float min_val = *std::min_element(domainB_images.begin(), domainB_images.end());
    float max_val = *std::max_element(domainB_images.begin(), domainB_images.end());
    float sum = std::accumulate(domainB_images.begin(), domainB_images.end(), 0.0f);
    float mean = sum / domainB_images.size();
    std::cout << "  Image range: [" << std::fixed << std::setprecision(3) 
              << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "  Image mean: " << mean << std::endl;
    
    // DomainA (기존 도메인) - 옵션
    auto domainA_images = loadNpy(domainA_images_path);
    auto domainA_labels = loadNpy(domainA_labels_path);
    
    bool has_domainA = !domainA_images.empty() && !domainA_labels.empty();
    if (has_domainA) {
        std::cout << "✓ DomainA loaded" << std::endl;
        std::cout << "  Images: " << domainA_images.size() << " floats" << std::endl;
        std::cout << "  Labels: " << domainA_labels.size() << " floats" << std::endl;
    } else {
        std::cout << "⚠ DomainA not found (will skip old domain evaluation)" << std::endl;
    }
    
    // ========================================
    // 학습 전 정확도 평가
    // ========================================
    std::cout << "\n--- Evaluating Before Training ---" << std::endl;
    
    auto* infer_runner = interpreter->GetSignatureRunner("infer");
    if (!infer_runner) {
        std::cerr << "Failed to get infer runner" << std::endl;
        return -1;
    }
    
    // Infer runner 초기화
    std::vector<int> infer_x_shape = {BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3};
    infer_runner->ResizeInputTensor("x", infer_x_shape);
    infer_runner->AllocateTensors();
    
    int numB_samples = domainB_images.size() / (IMG_SIZE * IMG_SIZE * 3);
    float acc_before_new = evaluateAccuracy(infer_runner, domainB_images, domainB_labels, numB_samples);
    
    std::cout << "Accuracy on new domain (before): " 
              << std::fixed << std::setprecision(2) << (acc_before_new * 100) << "%" << std::endl;
    
    float acc_before_old = 0.0f;
    if (has_domainA) {
        int numA_samples = domainA_images.size() / (IMG_SIZE * IMG_SIZE * 3);
        acc_before_old = evaluateAccuracy(infer_runner, domainA_images, domainA_labels, numA_samples);
        std::cout << "Accuracy on old domain (before): " 
                  << std::fixed << std::setprecision(2) << (acc_before_old * 100) << "%" << std::endl;
    }
    
    // ========================================
    // Exemplar 준비
    // ========================================
    std::cout << "\n--- Preparing Exemplars ---" << std::endl;
    ExemplarManager ex_mgr(EXEMPLARS_PER_CLASS);
    
    int actual_batch_size = domainB_images.size() / (IMG_SIZE * IMG_SIZE * 3);
    std::cout << "  Actual batch size: " << actual_batch_size << std::endl;
    
    for (int i = 0; i < 3; i++) {
        ex_mgr.addCandidates(domainB_images, domainB_labels, actual_batch_size);
    }
    
    std::cout << "Total exemplars: " << ex_mgr.getTotalSize() << std::endl;
    
    // ========================================
    // Train 준비
    // ========================================
    std::cout << "\n--- Preparing Train Runner ---" << std::endl;
    auto* train_runner = interpreter->GetSignatureRunner("train");
    if (!train_runner) {
        std::cerr << "Failed to get train runner" << std::endl;
        return -1;
    }

    std::vector<int> x_shape = {BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3};
    if (train_runner->ResizeInputTensor("x", x_shape) != kTfLiteOk) {
        std::cerr << "Failed to resize train input 'x'." << std::endl;
        return -1;
    }

    std::vector<int> y_shape = {BATCH_SIZE, NUM_CLASSES};
    if (train_runner->ResizeInputTensor("y", y_shape) != kTfLiteOk) {
        std::cerr << "Failed to resize train input 'y'." << std::endl;
        return -1;
    }

    if (train_runner->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors for train runner." << std::endl;
        return -1;
    }
    
    auto* x_tensor = train_runner->input_tensor("x");
    std::cout << "[Debug] Train 'x' bytes: " << x_tensor->bytes 
              << " (Expected: " << (BATCH_SIZE * IMG_SIZE * IMG_SIZE * 3 * sizeof(float)) << ")" << std::endl;

    std::cout << "✓ Train runner ready (Resized to Batch " << BATCH_SIZE << ")" << std::endl;
    
    // ========================================
    // 증분학습: Domain B (수정 - 전체 데이터 순환)
    // ========================================
    std::cout << "\n--- Training on Domain B (Real Data) ---" << std::endl;

    int total_samples = domainB_images.size() / (IMG_SIZE * IMG_SIZE * 3);
    int num_batches = (total_samples + BATCH_SIZE - 1) / BATCH_SIZE;

    std::cout << "  Total samples: " << total_samples << std::endl;
    std::cout << "  Batches per epoch: " << num_batches << std::endl;

    for (int step = 0; step < INCR_NEW_STEPS; step++) {
        int batch_idx = step % num_batches;
        int start_idx = batch_idx * BATCH_SIZE;
        int end_idx = std::min(start_idx + BATCH_SIZE, total_samples);
        int current_batch_size = end_idx - start_idx;
        
        // Dynamic resize if last batch is smaller
        if (current_batch_size != BATCH_SIZE) {
            std::vector<int> x_shape_dynamic = {current_batch_size, IMG_SIZE, IMG_SIZE, 3};
            std::vector<int> y_shape_dynamic = {current_batch_size, NUM_CLASSES};
            train_runner->ResizeInputTensor("x", x_shape_dynamic);
            train_runner->ResizeInputTensor("y", y_shape_dynamic);
            train_runner->AllocateTensors();
        }
        
        auto* x_tensor = train_runner->input_tensor("x");
        auto* y_tensor = train_runner->input_tensor("y");
        
        int img_size = IMG_SIZE * IMG_SIZE * 3;
        std::memcpy(x_tensor->data.f, 
                    domainB_images.data() + start_idx * img_size,
                    current_batch_size * img_size * sizeof(float));
        std::memcpy(y_tensor->data.f,
                    domainB_labels.data() + start_idx * NUM_CLASSES,
                    current_batch_size * NUM_CLASSES * sizeof(float));
        
        train_runner->Invoke();
        
        // Restore to standard batch size if changed
        if (current_batch_size != BATCH_SIZE) {
            std::vector<int> x_shape = {BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3};
            std::vector<int> y_shape = {BATCH_SIZE, NUM_CLASSES};
            train_runner->ResizeInputTensor("x", x_shape);
            train_runner->ResizeInputTensor("y", y_shape);
            train_runner->AllocateTensors();
        }
        
        auto* loss_tensor = train_runner->output_tensor("loss");
        float loss = loss_tensor->data.f[0];
        
        if (std::isnan(loss) || std::isinf(loss)) {
            std::cerr << "⚠ ERROR: Loss became NaN/Inf at domainB step " << (step + 1) << std::endl;
            std::cerr << "Stopping training to prevent corruption." << std::endl;
            return -1;
        }
        
        if ((step + 1) % 10 == 0) {
            std::cout << "  domainB step " << std::setw(3) << (step + 1) 
                    << " | loss: " << std::fixed << std::setprecision(4) << loss
                    << std::endl;
        }
    }
    
    // ========================================
    // 리허설
    // ========================================
    if (ex_mgr.getTotalSize() > 0) {
        std::cout << "\n--- Rehearsal Training ---" << std::endl;
        
        for (int step = 0; step < INCR_REHEARSAL_STEPS; step++) {
            std::vector<float> images, labels;
            
            if (!ex_mgr.getRehearsalBatch(images, labels, BATCH_SIZE)) {
                break;
            }
            
            int actual_batch_size = images.size() / (IMG_SIZE * IMG_SIZE * 3);
            
            if (actual_batch_size != BATCH_SIZE) {
                std::vector<int> x_shape_rehearsal = {actual_batch_size, IMG_SIZE, IMG_SIZE, 3};
                std::vector<int> y_shape_rehearsal = {actual_batch_size, NUM_CLASSES};
                
                train_runner->ResizeInputTensor("x", x_shape_rehearsal);
                train_runner->ResizeInputTensor("y", y_shape_rehearsal);
                train_runner->AllocateTensors();
            }
            
            auto* x_tensor = train_runner->input_tensor("x");
            auto* y_tensor = train_runner->input_tensor("y");
            
            std::memcpy(x_tensor->data.f, images.data(), images.size() * sizeof(float));
            std::memcpy(y_tensor->data.f, labels.data(), labels.size() * sizeof(float));
            
            train_runner->Invoke();
            
            if (actual_batch_size != BATCH_SIZE) {
                std::vector<int> x_shape = {BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3};
                std::vector<int> y_shape = {BATCH_SIZE, NUM_CLASSES};
                train_runner->ResizeInputTensor("x", x_shape);
                train_runner->ResizeInputTensor("y", y_shape);
                train_runner->AllocateTensors();
            }
            
            if ((step + 1) % 8 == 0) {
                auto* loss_tensor = train_runner->output_tensor("loss");
                float loss = loss_tensor->data.f[0];
                
                if (std::isnan(loss) || std::isinf(loss)) {
                    std::cerr << "⚠ WARNING: Loss became NaN/Inf at rehearsal step " << (step + 1) << std::endl;
                    break;
                }
                
                std::cout << "  rehearsal " << std::setw(3) << (step + 1)
                         << " | loss: " << std::fixed << std::setprecision(4) << loss
                         << std::endl;
            }
        }
    }
    
    // ========================================
    // 학습 후 정확도 평가
    // ========================================
    std::cout << "\n--- Evaluating After Training ---" << std::endl;
    
    std::cout << "Synchronizing weights..." << std::endl;
    auto* save_runner_sync = interpreter->GetSignatureRunner("save");
    if (!save_runner_sync) {
        std::cerr << "Failed to get save runner for sync" << std::endl;
        return -1;
    }
    
    save_runner_sync->AllocateTensors();
    save_runner_sync->Invoke();
    
    auto* synced_weights_tensor = save_runner_sync->output_tensor("weights");
    size_t num_synced = synced_weights_tensor->bytes / sizeof(float);
    std::vector<float> synced_weights(synced_weights_tensor->data.f, 
                                      synced_weights_tensor->data.f + num_synced);
    
    // Restore로 다시 적용 (infer runner가 최신 가중치를 볼 수 있도록)
    auto* restore_runner_sync = interpreter->GetSignatureRunner("restore");
    std::vector<int> sync_shape = { static_cast<int>(synced_weights.size()) };
    restore_runner_sync->ResizeInputTensor("flat_weights", sync_shape);
    restore_runner_sync->AllocateTensors();
    
    auto* sync_tensor = restore_runner_sync->input_tensor("flat_weights");
    std::memcpy(sync_tensor->data.f, synced_weights.data(), synced_weights.size() * sizeof(float));
    restore_runner_sync->Invoke();
    
    std::cout << "✓ Weights synchronized (" << num_synced << " floats)" << std::endl;
    
    // 이제 평가
    float acc_after_new = evaluateAccuracy(infer_runner, domainB_images, domainB_labels, numB_samples);
    
    std::cout << "Accuracy on new domain (after): " 
              << std::fixed << std::setprecision(2) << (acc_after_new * 100) << "%" << std::endl;
    
    float acc_after_old = acc_before_old;
    if (has_domainA) {
        int numA_samples = domainA_images.size() / (IMG_SIZE * IMG_SIZE * 3);
        acc_after_old = evaluateAccuracy(infer_runner, domainA_images, domainA_labels, numA_samples);
        std::cout << "Accuracy on old domain (after): " 
                  << std::fixed << std::setprecision(2) << (acc_after_old * 100) << "%" << std::endl;
    }
    
    // ========================================
    // Decision Signature 호출
    // ========================================
    std::cout << "\n--- Making Decision ---" << std::endl;

    auto* decision_runner = interpreter->GetSignatureRunner("decision");
    if (!decision_runner) {
        std::cerr << "Failed to get decision runner" << std::endl;
        return -1;
    }

    decision_runner->AllocateTensors();

    auto* acc_before_old_tensor = decision_runner->input_tensor("acc_before_old");
    auto* acc_before_new_tensor = decision_runner->input_tensor("acc_before_new");
    auto* acc_after_old_tensor = decision_runner->input_tensor("acc_after_old");
    auto* acc_after_new_tensor = decision_runner->input_tensor("acc_after_new");

    acc_before_old_tensor->data.f[0] = has_domainA ? acc_before_old : 0.0f;
    acc_before_new_tensor->data.f[0] = acc_before_new;
    acc_after_old_tensor->data.f[0] = has_domainA ? acc_after_old : 0.0f;
    acc_after_new_tensor->data.f[0] = acc_after_new;

    // Decision 실행
    decision_runner->Invoke();

    // 결과 확인
    auto* approve_tensor = decision_runner->output_tensor("approve");
    auto* retain_drop_tensor = decision_runner->output_tensor("retain_drop");
    auto* new_gain_tensor = decision_runner->output_tensor("new_gain");

    bool approve = approve_tensor->data.b[0];
    float retain_drop = retain_drop_tensor->data.f[0];
    float new_gain = new_gain_tensor->data.f[0];

    std::cout << "\nDecision Results:" << std::endl;
    if (has_domainA) {
        std::cout << "  Old domain (before): " << std::fixed << std::setprecision(2) 
                << (acc_before_old * 100) << "%" << std::endl;
        std::cout << "  Old domain (after):  " << (acc_after_old * 100) << "%" << std::endl;
    }
    std::cout << "  New domain (before): " << (acc_before_new * 100) << "%" << std::endl;
    std::cout << "  New domain (after):  " << (acc_after_new * 100) << "%" << std::endl;
    std::cout << "\n  New domain gain: " << std::fixed << std::setprecision(4) 
            << new_gain << " (+" << (new_gain * 100) << "%)" << std::endl;
    std::cout << "  Old domain drop: " << retain_drop 
            << " (-" << (retain_drop * 100) << "%)" << std::endl;
    std::cout << "  Threshold: gain >= 3%, drop <= 2%" << std::endl;
    std::cout << "  Approved: " << (approve ? "YES ✓" : "NO ✗") << std::endl;

    // ========================================
    // Decision
    // ========================================
    if (approve) {
        std::cout << "\n--- Save Updated Checkpoint (Approved) ---" << std::endl;
        auto* save_runner = interpreter->GetSignatureRunner("save");
        if (!save_runner) {
            std::cerr << "Failed to get save runner" << std::endl;
            return -1;
        }
        save_runner->AllocateTensors();
        save_runner->Invoke();
        
        auto* saved_weights = save_runner->output_tensor("weights");
        size_t num_weights = saved_weights->bytes / sizeof(float);
        std::vector<float> updated_weights(saved_weights->data.f, 
                                          saved_weights->data.f + num_weights);
        
        saveNpy(ckpt_after, updated_weights);
        std::cout << "✓ Update approved and saved" << std::endl;
    } else {
        std::cout << "\n--- Update Rejected ---" << std::endl;
        std::cout << "✗ Checkpoint not saved. Performance criteria not met." << std::endl;
        
        if (new_gain < 0.03) {
            std::cout << "  Reason: New domain gain (" << (new_gain * 100) 
                     << "%) is below threshold (3%)" << std::endl;
        }
        if (retain_drop > 0.02) {
            std::cout << "  Reason: Old domain drop (" << (retain_drop * 100) 
                     << "%) exceeds threshold (2%)" << std::endl;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Phase-2 Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
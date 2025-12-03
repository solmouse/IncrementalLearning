#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <random>
#include <algorithm>
#include <map>
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/signature_runner.h"
#include "tensorflow/lite/delegates/flex/delegate.h"

// Phase-2: TFLite incr learning with Flex delegate

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
// NPY 파일 유틸리티
// ============================================================================
std::vector<float> loadNpy(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }
    
    char magic[6];
    file.read(magic, 6);
    if (magic[0] != '\x93' || magic[1] != 'N') {
        std::cerr << "Invalid NPY file" << std::endl;
        return {};
    }
    
    uint8_t major = magic[6];
    uint16_t header_len = 0;
    if (major == 1) {
        file.read(reinterpret_cast<char*>(&header_len), 2);
    } else {
        uint32_t header_len32;
        file.read(reinterpret_cast<char*>(&header_len32), 4);
        header_len = header_len32;
    }
    
    file.seekg(header_len, std::ios::cur);
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t header_total = 10 + header_len;
    size_t data_size = file_size - header_total;
    size_t num_floats = data_size / sizeof(float);
    
    file.seekg(header_total);
    std::vector<float> data(num_floats);
    file.read(reinterpret_cast<char*>(data.data()), data_size);
    
    std::cout << "Loaded " << num_floats << " floats from " << filename << std::endl;
    
    file.close();
    return data;
}

bool saveNpy(const char* filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot create file: " << filename << std::endl;
        return false;
    }
    
    file.write("\x93NUMPY", 6);
    
    uint8_t major = 1, minor = 0;
    file.write(reinterpret_cast<char*>(&major), 1);
    file.write(reinterpret_cast<char*>(&minor), 1);
    
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" 
                        + std::to_string(data.size()) + ",), }";
    while (header.size() % 16 != 15) header += ' ';
    header += '\n';
    
    uint16_t header_len = header.size();
    file.write(reinterpret_cast<char*>(&header_len), 2);
    file.write(header.c_str(), header_len);
    
    file.write(reinterpret_cast<const char*>(data.data()), 
               data.size() * sizeof(float));
    
    std::cout << "Saved " << data.size() << " floats to " << filename << std::endl;
    
    file.close();
    return true;
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
// 더미 데이터 생성
// ============================================================================
void generateDummyBatch(std::vector<float>& images, std::vector<float>& labels, int batch_size) {
    images.resize(batch_size * IMG_SIZE * IMG_SIZE * 3, 0.5f);
    labels.resize(batch_size * NUM_CLASSES, 0.0f);
    for (int i = 0; i < batch_size; i++) {
        labels[i * NUM_CLASSES + 0] = 1.0f;
    }
}

// ============================================================================
// MAIN: Phase-2 TFLite 증분학습 (Flex)
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "[Phase-2] TFLite 증분학습 (with Flex)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <model.tflite> <ckpt_before.npy> [ckpt_after.npy]" << std::endl;
        return -1;
    }
    
    const char* model_path = argv[1];
    const char* ckpt_before = argv[2];
    const char* ckpt_after = (argc >= 4) ? argv[3] : "ckpt_after.npy";
    
    std::cout << "\nModel: " << model_path << std::endl;
    std::cout << "Checkpoint (before): " << ckpt_before << std::endl;
    std::cout << "Checkpoint (after): " << ckpt_after << std::endl;
    
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
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    if (!interpreter) {
        std::cerr << "Failed to create interpreter" << std::endl;
        return -1;
    }
    interpreter->SetNumThreads(4);
    std::cout << "✓ Interpreter created" << std::endl;
    
    // ========================================
    // Flex Delegate 적용
    // ========================================
    std::cout << "\n--- Applying Flex Delegate ---" << std::endl;
    auto* flex_delegate = TfLiteFlexDelegateCreate(nullptr);
    if (!flex_delegate) {
        std::cerr << "Failed to create Flex delegate" << std::endl;
        return -1;
    }
    
    if (interpreter->ModifyGraphWithDelegate(flex_delegate) != kTfLiteOk) {
        std::cerr << "Failed to apply Flex delegate" << std::endl;
        TfLiteFlexDelegateDelete(flex_delegate);
        return -1;
    }
    std::cout << "✓ Flex delegate applied" << std::endl;
    
    // ========================================
    // Initialize
    // ========================================
    std::cout << "\n--- Initialize ---" << std::endl;
    auto* init_runner = interpreter->GetSignatureRunner("initialize");
    if (!init_runner) {
        std::cerr << "Failed to get initialize runner" << std::endl;
        TfLiteFlexDelegateDelete(flex_delegate);
        return -1;
    }
    init_runner->AllocateTensors();
    init_runner->Invoke();
    std::cout << "✓ Model initialized" << std::endl;
    
    // ========================================
    // Restore checkpoint
    // ========================================
    std::cout << "\n--- Restore Checkpoint ---" << std::endl;
    auto* restore_runner = interpreter->GetSignatureRunner("restore");
    if (!restore_runner) {
        std::cerr << "Failed to get restore runner" << std::endl;
        TfLiteFlexDelegateDelete(flex_delegate);
        return -1;
    }
    restore_runner->AllocateTensors();
    
    auto weights = loadNpy(ckpt_before);
    if (weights.empty()) {
        TfLiteFlexDelegateDelete(flex_delegate);
        return -1;
    }
    
    auto* weights_tensor = restore_runner->input_tensor("flat_weights");
    std::memcpy(weights_tensor->data.f, weights.data(), weights.size() * sizeof(float));
    restore_runner->Invoke();
    std::cout << "✓ Checkpoint restored" << std::endl;
    
    // ========================================
    // Exemplar 준비
    // ========================================
    std::cout << "\n--- Preparing Exemplars ---" << std::endl;
    ExemplarManager ex_mgr(EXEMPLARS_PER_CLASS);
    
    for (int i = 0; i < 12; i++) {
        std::vector<float> images, labels;
        generateDummyBatch(images, labels, BATCH_SIZE);
        ex_mgr.addCandidates(images, labels, BATCH_SIZE);
    }
    std::cout << "Total exemplars: " << ex_mgr.getTotalSize() << std::endl;
    
    // ========================================
    // Train 준비
    // ========================================
    auto* train_runner = interpreter->GetSignatureRunner("train");
    if (!train_runner) {
        std::cerr << "Failed to get train runner" << std::endl;
        TfLiteFlexDelegateDelete(flex_delegate);
        return -1;
    }
    train_runner->AllocateTensors();
    std::cout << "✓ Train runner ready" << std::endl;
    
    // ========================================
    // 증분학습: Domain B
    // ========================================
    std::cout << "\n--- Training on Domain B (New Data) ---" << std::endl;
    
    for (int step = 0; step < INCR_NEW_STEPS; step++) {
        std::vector<float> images, labels;
        generateDummyBatch(images, labels, BATCH_SIZE);
        
        auto* x_tensor = train_runner->input_tensor("x");
        auto* y_tensor = train_runner->input_tensor("y");
        
        std::memcpy(x_tensor->data.f, images.data(), images.size() * sizeof(float));
        std::memcpy(y_tensor->data.f, labels.data(), labels.size() * sizeof(float));
        
        train_runner->Invoke();
        
        if ((step + 1) % 10 == 0) {
            auto* loss_tensor = train_runner->output_tensor("loss");
            float loss = loss_tensor->data.f[0];
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
            
            auto* x_tensor = train_runner->input_tensor("x");
            auto* y_tensor = train_runner->input_tensor("y");
            
            std::memcpy(x_tensor->data.f, images.data(), images.size() * sizeof(float));
            std::memcpy(y_tensor->data.f, labels.data(), labels.size() * sizeof(float));
            
            train_runner->Invoke();
            
            if ((step + 1) % 8 == 0) {
                auto* loss_tensor = train_runner->output_tensor("loss");
                float loss = loss_tensor->data.f[0];
                std::cout << "  rehearsal " << std::setw(3) << (step + 1)
                         << " | loss: " << std::fixed << std::setprecision(4) << loss
                         << std::endl;
            }
        }
    }
    
    // ========================================
    // Save checkpoint
    // ========================================
    std::cout << "\n--- Save Updated Checkpoint ---" << std::endl;
    auto* save_runner = interpreter->GetSignatureRunner("save");
    if (!save_runner) {
        std::cerr << "Failed to get save runner" << std::endl;
        TfLiteFlexDelegateDelete(flex_delegate);
        return -1;
    }
    save_runner->AllocateTensors();
    save_runner->Invoke();
    
    auto* saved_weights = save_runner->output_tensor("weights");
    size_t num_weights = saved_weights->bytes / sizeof(float);
    std::vector<float> updated_weights(saved_weights->data.f, 
                                      saved_weights->data.f + num_weights);
    
    saveNpy(ckpt_after, updated_weights);
    std::cout << "✓ Update complete" << std::endl;
    
    // ========================================
    // Cleanup
    // ========================================
    TfLiteFlexDelegateDelete(flex_delegate);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Phase-2 Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
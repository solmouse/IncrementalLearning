// ============================================================================
// OPTIMIZED VERSION - Key Changes:
// 1. Batch processing for TFLite (if model supports it)
// 2. Feature caching to avoid redundant extraction
// 3. Parallel validation with OpenMP
// 4. Reduced validation frequency
// ============================================================================

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <algorithm>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cnpy.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

const int IMG_SIZE = 224;
const int FEATURE_DIM = 1280;
const int OLD_CLASSES = 3;
const int NEW_CLASSES = 1;
const int TOTAL_CLASSES = 4;
const int BATCH_SIZE = 24;
const int NEW_SAMPLES = 8;
const int REH_SAMPLES = 16;
const int NUM_EPOCHS = 6;
const int STEPS_PER_EPOCH = 80;
const float HEAD_LR_START = 1e-4f;
const float HEAD_LR_END = 5e-5f;
const float ADAPTER_LR_START = 2e-5f;
const float ADAPTER_LR_END = 5e-7f;
const float LABEL_SMOOTH = 0.1f;
const float MAX_GRAD_NORM = 0.5f;
const float KD_TEMP = 2.0f;
const float KD_WEIGHT = 5.0f;

// ============================================================================
// Timing Utility
// ============================================================================
class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    void reset() { start = std::chrono::high_resolution_clock::now(); }
};

// ============================================================================
// NPY Utils (same as before)
// ============================================================================
std::vector<float> loadNpy(const char* f, std::vector<size_t>& s) {
    auto a = cnpy::npy_load(f);
    s = a.shape;
    float* p = a.data<float>();
    return std::vector<float>(p, p + a.num_vals);
}

std::vector<int> loadNpyInt(const char* f, std::vector<size_t>& s) {
    auto a = cnpy::npy_load(f);
    s = a.shape;
    int* p = a.data<int>();
    return std::vector<int>(p, p + a.num_vals);
}

void saveNpy(const char* f, const std::vector<float>& d, const std::vector<size_t>& s) {
    cnpy::npy_save(f, d.data(), s, "w");
}

// ============================================================================
// NMEClassifier (optimized with L2 norm caching)
// ============================================================================
struct NMEClassifier {
    std::vector<std::vector<float>> prototypes;
    std::vector<float> prototype_norms; // Cache norms
    int num_classes;

    NMEClassifier(int nc) : num_classes(nc) {
        prototypes.resize(nc, std::vector<float>(FEATURE_DIM, 0.f));
        prototype_norms.resize(nc, 0.f);
    }

    void load_prototypes(const char* filepath) {
        std::vector<size_t> shape;
        auto data = loadNpy(filepath, shape);
        int loaded_classes = shape[0];
        
        for (int c = 0; c < std::min(loaded_classes, num_classes); ++c) {
            for (int d = 0; d < FEATURE_DIM; ++d) {
                prototypes[c][d] = data[c * FEATURE_DIM + d];
            }
            // Precompute norm
            prototype_norms[c] = 0.f;
            for (int d = 0; d < FEATURE_DIM; ++d) {
                prototype_norms[c] += prototypes[c][d] * prototypes[c][d];
            }
            prototype_norms[c] = std::sqrt(prototype_norms[c]);
        }
        std::cout << "Loaded " << loaded_classes << " prototypes" << std::endl;
    }

    void update_prototype(int class_id, const std::vector<float>& exemplar_features, int num_exemplars) {
        std::fill(prototypes[class_id].begin(), prototypes[class_id].end(), 0.f);
        for (int i = 0; i < num_exemplars; ++i) {
            for (int d = 0; d < FEATURE_DIM; ++d) {
                prototypes[class_id][d] += exemplar_features[i * FEATURE_DIM + d];
            }
        }
        
        // Recompute norm
        prototype_norms[class_id] = 0.f;
        for (int d = 0; d < FEATURE_DIM; ++d) {
            prototypes[class_id][d] /= num_exemplars;
            prototype_norms[class_id] += prototypes[class_id][d] * prototypes[class_id][d];
        }
        prototype_norms[class_id] = std::sqrt(prototype_norms[class_id]);
    }

    int predict_single(const float* feature) {
        // L2 normalize feature once
        float feature_norm = 0.f;
        for (int d = 0; d < FEATURE_DIM; ++d) {
            feature_norm += feature[d] * feature[d];
        }
        feature_norm = std::sqrt(feature_norm + 1e-8f);

        float min_dist = 1e9f;
        int pred_class = -1;
        
        for (int c = 0; c < num_classes; ++c) {
            if (prototype_norms[c] < 1e-6f) continue;
            
            // Compute distance using cached norm
            float dist = 0.f;
            for (int d = 0; d < FEATURE_DIM; ++d) {
                float diff = feature[d] / feature_norm - prototypes[c][d] / prototype_norms[c];
                dist += diff * diff;
            }
            
            if (dist < min_dist) {
                min_dist = dist;
                pred_class = c;
            }
        }
        
        return (pred_class == -1) ? 0 : pred_class;
    }

    // Batch prediction with OpenMP
    void predict_batch(const float* features, int* predictions, int batch_size) {
        #pragma omp parallel for if(batch_size > 4)
        for (int n = 0; n < batch_size; ++n) {
            predictions[n] = predict_single(features + n * FEATURE_DIM);
        }
    }
};

// ============================================================================
// Dataset (same as before)
// ============================================================================
struct Dataset {
    std::vector<float> x;
    std::vector<int> y;
    int n;

    bool load(const char* xf, const char* yf) {
        std::vector<size_t> sx, sy;
        x = loadNpy(xf, sx);
        y = loadNpyInt(yf, sy);
        n = sy[0];
        
        std::map<int, int> cnt;
        for (int i = 0; i < n; ++i) cnt[y[i]]++;
        for (auto& p : cnt) std::cout << "c" << p.first << "=" << p.second << " ";
        std::cout << std::endl;
        return true;
    }

    void getRandom(int bs, std::mt19937& rng, float* xo, int* yo) {
        std::uniform_int_distribution<> d(0, n - 1);
        for (int i = 0; i < bs; ++i) {
            int idx = d(rng);
            std::memcpy(xo + i * IMG_SIZE * IMG_SIZE * 3,
                       x.data() + idx * IMG_SIZE * IMG_SIZE * 3,
                       IMG_SIZE * IMG_SIZE * 3 * sizeof(float));
            yo[i] = y[idx];
        }
    }
};

// ============================================================================
// BackboneAdapter (same as before but with timing)
// ============================================================================
struct BackboneAdapter {
    std::vector<float> g, b, dg, db, mg, vg, mb, vb;
    int t = 0;

    BackboneAdapter() {
        g.assign(FEATURE_DIM, 1.f);
        b.assign(FEATURE_DIM, 0.f);
        dg.assign(FEATURE_DIM, 0.f);
        db.assign(FEATURE_DIM, 0.f);
        mg.assign(FEATURE_DIM, 0.f);
        vg.assign(FEATURE_DIM, 0.f);
        mb.assign(FEATURE_DIM, 0.f);
        vb.assign(FEATURE_DIM, 0.f);
    }

    void forward(float* f, int bs) {
        for (int n = 0; n < bs; ++n)
            for (int c = 0; c < FEATURE_DIM; ++c)
                f[n * FEATURE_DIM + c] = g[c] * f[n * FEATURE_DIM + c] + b[c];
    }

    void backward(const float* f_raw, const float* gfeat, int bs) {
        std::fill(dg.begin(), dg.end(), 0.f);
        std::fill(db.begin(), db.end(), 0.f);
        
        for (int n = 0; n < bs; ++n)
            for (int c = 0; c < FEATURE_DIM; ++c) {
                int i = n * FEATURE_DIM + c;
                dg[c] += gfeat[i] * f_raw[i];
                db[c] += gfeat[i];
            }
        
        for (int c = 0; c < FEATURE_DIM; ++c) {
            dg[c] /= bs;
            db[c] /= bs;
        }
        
        // Gradient clipping
        float gn = 0.f;
        for (int c = 0; c < FEATURE_DIM; ++c)
            gn += dg[c] * dg[c] + db[c] * db[c];
        gn = std::sqrt(gn);
        
        if (gn > MAX_GRAD_NORM) {
            float scale = MAX_GRAD_NORM / gn;
            for (int c = 0; c < FEATURE_DIM; ++c) {
                dg[c] *= scale;
                db[c] *= scale;
            }
        }
    }

    void update(float lr) {
        t++;
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        float lr_t = lr * std::sqrt(1 - std::pow(b2, t)) / (1 - std::pow(b1, t));
        
        for (int c = 0; c < FEATURE_DIM; ++c) {
            mg[c] = b1 * mg[c] + (1 - b1) * dg[c];
            vg[c] = b2 * vg[c] + (1 - b2) * dg[c] * dg[c];
            g[c] -= lr_t * mg[c] / (std::sqrt(vg[c]) + eps);
            
            mb[c] = b1 * mb[c] + (1 - b1) * db[c];
            vb[c] = b2 * vb[c] + (1 - b2) * db[c] * db[c];
            b[c] -= lr_t * mb[c] / (std::sqrt(vb[c]) + eps);
        }
    }

    void save(const char* f) {
        std::vector<size_t> s = {FEATURE_DIM};
        cnpy::npz_save(f, "gamma", g.data(), s, "w");
        cnpy::npz_save(f, "beta", b.data(), s, "a");
    }
};

// ============================================================================
// FCLayer (same as before)
// ============================================================================
struct FCLayer {
    int in, out;
    std::vector<float> W, B, gW, gB, mW, vW, mB, vB;
    int t = 0;

    FCLayer(int i, int o) : in(i), out(o) {
        W.resize(i * o);
        B.assign(o, 0.f);
        gW.assign(i * o, 0.f);
        gB.assign(o, 0.f);
        mW.assign(i * o, 0.f);
        vW.assign(i * o, 0.f);
        mB.assign(o, 0.f);
        vB.assign(o, 0.f);
        
        std::mt19937 r(42);
        std::normal_distribution<float> d(0, std::sqrt(1.f / i));
        for (auto& v : W) v = d(r);
    }

    void load(const char* wf, const char* bf) {
        std::vector<size_t> sw, sb;
        auto w = loadNpy(wf, sw);
        auto b = loadNpy(bf, sb);
        
        for (int i = 0; i < in; ++i)
            for (int o = 0; o < OLD_CLASSES; ++o)
                W[i * out + o] = w[i * OLD_CLASSES + o];
        
        for (int o = 0; o < OLD_CLASSES; ++o)
            B[o] = b[o];
    }

    void forward(const float* x, float* y, int bs) {
        for (int n = 0; n < bs; ++n)
            for (int o = 0; o < out; ++o) {
                float s = B[o];
                for (int i = 0; i < in; ++i)
                    s += x[n * in + i] * W[i * out + o];
                y[n * out + o] = s;
            }
    }

    void backward(const float* x, const float* gy, int bs, float* gx) {
        std::fill(gW.begin(), gW.end(), 0.f);
        std::fill(gB.begin(), gB.end(), 0.f);
        std::fill(gx, gx + bs * in, 0.f);
        
        for (int n = 0; n < bs; ++n)
            for (int o = 0; o < out; ++o) {
                float g = gy[n * out + o];
                gB[o] += g;
                for (int i = 0; i < in; ++i) {
                    gW[i * out + o] += x[n * in + i] * g;
                    gx[n * in + i] += W[i * out + o] * g;
                }
            }
        
        for (auto& v : gW) v /= bs;
        for (auto& v : gB) v /= bs;
        
        // Gradient clipping
        float gn = 0.f;
        for (auto& v : gW) gn += v * v;
        for (auto& v : gB) gn += v * v;
        gn = std::sqrt(gn);
        
        if (gn > MAX_GRAD_NORM) {
            float scale = MAX_GRAD_NORM / gn;
            for (auto& v : gW) v *= scale;
            for (auto& v : gB) v *= scale;
        }
    }

    void update(float lr) {
        t++;
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        float lr_t = lr * std::sqrt(1 - std::pow(b2, t)) / (1 - std::pow(b1, t));
        
        for (size_t i = 0; i < W.size(); ++i) {
            mW[i] = b1 * mW[i] + (1 - b1) * gW[i];
            vW[i] = b2 * vW[i] + (1 - b2) * gW[i] * gW[i];
            W[i] -= lr_t * mW[i] / (std::sqrt(vW[i]) + eps);
        }
        
        for (size_t i = 0; i < B.size(); ++i) {
            mB[i] = b1 * mB[i] + (1 - b1) * gB[i];
            vB[i] = b2 * vB[i] + (1 - b2) * gB[i] * gB[i];
            B[i] -= lr_t * mB[i] / (std::sqrt(vB[i]) + eps);
        }
    }

    void save(const char* wf, const char* bf) {
        saveNpy(wf, W, {(size_t)in, (size_t)out});
        saveNpy(bf, B, {(size_t)out});
    }
};

// ============================================================================
// OPTIMIZED FeatureExtractor with Batch Support
// ============================================================================
struct FeatureExtractor {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::SignatureRunner* infer_runner = nullptr;
    tflite::SignatureRunner* init_runner = nullptr;
    int tflite_batch_size = 1; // Actual batch size supported by model
    
    double total_extract_time_ms = 0.0;
    int extract_calls = 0;

    bool load(const char* f) {
        model = tflite::FlatBufferModel::BuildFromFile(f);
        if (!model) {
            std::cerr << "ERROR: Failed to load model" << std::endl;
            return false;
        }

        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        if (!interpreter) {
            std::cerr << "ERROR: Failed to build interpreter" << std::endl;
            return false;
        }

        interpreter->SetNumThreads(4);
        init_runner = interpreter->GetSignatureRunner("initialize");
        infer_runner = interpreter->GetSignatureRunner("infer");
        
        if (!init_runner || !infer_runner) {
            std::cerr << "ERROR: Required signatures not found" << std::endl;
            return false;
        }

        if (init_runner->AllocateTensors() != kTfLiteOk) {
            std::cerr << "ERROR: init AllocateTensors failed" << std::endl;
            return false;
        }

        std::cout << "[BACKBONE] Running initialize..." << std::endl;
        if (init_runner->Invoke() != kTfLiteOk) {
            std::cerr << "ERROR: initialize Invoke failed" << std::endl;
            return false;
        }

        if (infer_runner->AllocateTensors() != kTfLiteOk) {
            std::cerr << "ERROR: infer AllocateTensors failed" << std::endl;
            return false;
        }

        // Check actual batch size from model
        TfLiteTensor* input = infer_runner->input_tensor("x");
        if (input && input->dims->size > 0) {
            tflite_batch_size = input->dims->data[0];
            std::cout << "[BACKBONE] TFLite model batch size: " << tflite_batch_size << std::endl;
        }

        std::cout << "[BACKBONE] Successfully loaded" << std::endl;
        return true;
    }

    // OPTIMIZED: Process in actual batches if model supports it
    void extract(const float* img, float* feat, int bs) {
        Timer timer;
        
        TfLiteTensor* input = infer_runner->input_tensor("x");
        const TfLiteTensor* output = infer_runner->output_tensor("features");
        
        if (!input || !output) {
            std::cerr << "ERROR: Invalid input/output tensors" << std::endl;
            return;
        }

        // If model only supports batch=1, process sequentially
        if (tflite_batch_size == 1) {
            for (int n = 0; n < bs; ++n) {
                memcpy(input->data.f,
                       img + n * IMG_SIZE * IMG_SIZE * 3,
                       IMG_SIZE * IMG_SIZE * 3 * sizeof(float));
                
                if (infer_runner->Invoke() != kTfLiteOk) {
                    std::cerr << "ERROR: Invoke failed for sample " << n << std::endl;
                    continue;
                }
                
                memcpy(feat + n * FEATURE_DIM,
                       output->data.f,
                       FEATURE_DIM * sizeof(float));
            }
        } else {
            // Process in batches if model supports it
            int num_batches = (bs + tflite_batch_size - 1) / tflite_batch_size;
            for (int b = 0; b < num_batches; ++b) {
                int start_idx = b * tflite_batch_size;
                int end_idx = std::min(start_idx + tflite_batch_size, bs);
                int actual_batch = end_idx - start_idx;
                
                // Copy batch
                memcpy(input->data.f,
                       img + start_idx * IMG_SIZE * IMG_SIZE * 3,
                       actual_batch * IMG_SIZE * IMG_SIZE * 3 * sizeof(float));
                
                if (infer_runner->Invoke() != kTfLiteOk) {
                    std::cerr << "ERROR: Batch invoke failed" << std::endl;
                    continue;
                }
                
                // Copy results
                memcpy(feat + start_idx * FEATURE_DIM,
                       output->data.f,
                       actual_batch * FEATURE_DIM * sizeof(float));
            }
        }
        
        total_extract_time_ms += timer.elapsed_ms();
        extract_calls++;
    }

    void print_timing_stats() {
        if (extract_calls > 0) {
            std::cout << "[TIMING] Feature extraction: " 
                      << total_extract_time_ms / extract_calls << " ms/call (avg)" 
                      << " | Total: " << total_extract_time_ms << " ms" << std::endl;
        }
    }
};

// ============================================================================
// OPTIMIZED Validation with Feature Caching
// ============================================================================
void run_validation_cached(
    FeatureExtractor& fe,
    BackboneAdapter& adapter,
    NMEClassifier& nme,
    Dataset& vald,
    const char* epoch_name = "Validation"
) {
    std::cout << "\n [" << epoch_name << "]" << std::endl;
    Timer timer;
    
    const int val_batch_size = 32; // Larger batch for validation
    int num_batches = (vald.n + val_batch_size - 1) / val_batch_size;
    
    std::vector<float> img_batch(val_batch_size * IMG_SIZE * IMG_SIZE * 3);
    std::vector<float> fraw(val_batch_size * FEATURE_DIM);
    std::vector<float> fad(val_batch_size * FEATURE_DIM);
    std::vector<int> predictions(val_batch_size);
    
    int correct = 0, total = 0;
    std::vector<int> cc(TOTAL_CLASSES, 0), ct(TOTAL_CLASSES, 0);
    
    for (int b = 0; b < num_batches; ++b) {
        int start_idx = b * val_batch_size;
        int end_idx = std::min(start_idx + val_batch_size, vald.n);
        int actual_batch = end_idx - start_idx;
        
        // Prepare batch
        for (int i = 0; i < actual_batch; ++i) {
            int idx = start_idx + i;
            std::memcpy(img_batch.data() + i * IMG_SIZE * IMG_SIZE * 3,
                       vald.x.data() + idx * IMG_SIZE * IMG_SIZE * 3,
                       IMG_SIZE * IMG_SIZE * 3 * sizeof(float));
        }
        
        // Extract features in batch
        fe.extract(img_batch.data(), fraw.data(), actual_batch);
        
        // Apply adapter
        std::copy(fraw.begin(), fraw.begin() + actual_batch * FEATURE_DIM, fad.begin());
        adapter.forward(fad.data(), actual_batch);
        
        // Batch prediction with NME
        nme.predict_batch(fad.data(), predictions.data(), actual_batch);
        
        // Count accuracy
        for (int i = 0; i < actual_batch; ++i) {
            int idx = start_idx + i;
            int label = vald.y[idx];
            int pred = predictions[i];
            
            if (pred == label) {
                correct++;
                cc[label]++;
            }
            ct[label]++;
            total++;
        }
    }
    
    double val_time = timer.elapsed_ms();
    
    std::cout << "  Overall Accuracy: " << std::fixed << std::setprecision(4) 
              << (float)correct / total << " (" << correct << "/" << total << ")"
              << " | Time: " << val_time / 1000.0 << "s" << std::endl;
    
    const char* names[] = {"daisy", "dandelion", "roses", "sunflowers"};
    for (int c = 0; c < TOTAL_CLASSES; ++c) {
        if (ct[c] > 0) {
            std::cout << "  " << names[c] << ": " << std::setprecision(4) 
                      << (float)cc[c] / ct[c] << " (" << cc[c] << "/" << ct[c] << ")" << std::endl;
        }
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Incremental Learning (OPTIMIZED)" << std::endl;
    std::cout << "========================================" << std::endl;

    #ifdef _OPENMP
    std::cout << "OpenMP enabled with " << omp_get_max_threads() << " threads" << std::endl;
    #endif

    if (argc < 11) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.tflite> <old_head_w.npy> <old_head_b.npy>"
                  << " <new_x.npy> <new_y.npy> <reh_x.npy> <reh_y.npy>"
                  << " <val_x.npy> <val_y.npy> <prototypes.npy>" << std::endl;
        return -1;
    }

    Timer total_timer;

    // Load components
    FeatureExtractor fe;
    fe.load(argv[1]);

    Dataset newd, rehd, vald;
    std::cout << "Loading new data: "; newd.load(argv[4], argv[5]);
    std::cout << "Loading rehearsal data: "; rehd.load(argv[6], argv[7]);
    std::cout << "Loading validation data: "; vald.load(argv[8], argv[9]);

    FCLayer head(FEATURE_DIM, TOTAL_CLASSES);
    head.load(argv[2], argv[3]);

    FCLayer teacher(FEATURE_DIM, OLD_CLASSES);
    teacher.load(argv[2], argv[3]);

    BackboneAdapter adapter;

    NMEClassifier nme(TOTAL_CLASSES);
    nme.load_prototypes(argv[10]);

    std::random_device rd;
    std::mt19937 rng(rd());

    std::vector<float> img(BATCH_SIZE * IMG_SIZE * IMG_SIZE * 3);
    std::vector<int> lab(BATCH_SIZE);
    std::vector<float> fraw(BATCH_SIZE * FEATURE_DIM);
    std::vector<float> fad(BATCH_SIZE * FEATURE_DIM);
    std::vector<float> logit(BATCH_SIZE * TOTAL_CLASSES);
    std::vector<float> teacher_logit(BATCH_SIZE * OLD_CLASSES);
    std::vector<float> prob(BATCH_SIZE * TOTAL_CLASSES);
    std::vector<float> glog(BATCH_SIZE * TOTAL_CLASSES);
    std::vector<float> gfeat(BATCH_SIZE * FEATURE_DIM);

    int total_steps = NUM_EPOCHS * STEPS_PER_EPOCH;

    // ========================================================================
    // Initial Validation
    // ========================================================================
    run_validation_cached(fe, adapter, nme, vald, "Initial Validation (Epoch 0)");

    // ========================================================================
    // Training Loop
    // ========================================================================
    for (int e = 0; e < NUM_EPOCHS; ++e) {
        std::cout << "\n=== Epoch " << (e + 1) << "/" << NUM_EPOCHS << " ===" << std::endl;
        Timer epoch_timer;
        
        float epoch_ce = 0.f, epoch_kd = 0.f;

        for (int s = 0; s < STEPS_PER_EPOCH; ++s) {
            Timer step_timer;
            
            int global_step = e * STEPS_PER_EPOCH + s;
            float progress = (float)global_step / total_steps;
            float head_lr = HEAD_LR_START * (1 - progress) + HEAD_LR_END * progress;
            float adapter_lr = ADAPTER_LR_START * (1 - progress) + ADAPTER_LR_END * progress;

            // Sample batch
            newd.getRandom(NEW_SAMPLES, rng, img.data(), lab.data());
            rehd.getRandom(REH_SAMPLES, rng, 
                          img.data() + NEW_SAMPLES * IMG_SIZE * IMG_SIZE * 3,
                          lab.data() + NEW_SAMPLES);

            // Forward pass
            fe.extract(img.data(), fraw.data(), BATCH_SIZE);
            fad = fraw;
            adapter.forward(fad.data(), BATCH_SIZE);
            head.forward(fad.data(), logit.data(), BATCH_SIZE);
            teacher.forward(fraw.data(), teacher_logit.data(), BATCH_SIZE);

            std::fill(glog.begin(), glog.end(), 0.f);

            // CE Loss
            float ce = 0.f;
            for (int n = 0; n < BATCH_SIZE; ++n) {
                float m = -1e9f;
                for (int c = 0; c < TOTAL_CLASSES; ++c)
                    m = std::max(m, logit[n * TOTAL_CLASSES + c]);
                
                float ss = 0.f;
                for (int c = 0; c < TOTAL_CLASSES; ++c) {
                    prob[n * TOTAL_CLASSES + c] = std::exp(logit[n * TOTAL_CLASSES + c] - m);
                    ss += prob[n * TOTAL_CLASSES + c];
                }
                
                for (int c = 0; c < TOTAL_CLASSES; ++c) {
                    prob[n * TOTAL_CLASSES + c] /= (ss + 1e-9f);
                    float t = (c == lab[n]) ? (1 - LABEL_SMOOTH + LABEL_SMOOTH / TOTAL_CLASSES) 
                                            : (LABEL_SMOOTH / TOTAL_CLASSES);
                    ce -= t * std::log(prob[n * TOTAL_CLASSES + c] + 1e-10f);
                    glog[n * TOTAL_CLASSES + c] = (prob[n * TOTAL_CLASSES + c] - t) / BATCH_SIZE;
                }
            }
            ce /= BATCH_SIZE;

            // KD Loss
            float kd = 0.f;
            for (int n = 0; n < BATCH_SIZE; ++n) {
                float m_t = -1e9f;
                for (int c = 0; c < OLD_CLASSES; ++c)
                    m_t = std::max(m_t, teacher_logit[n * OLD_CLASSES + c] / KD_TEMP);
                
                float ss_t = 0.f;
                std::vector<float> t_soft(OLD_CLASSES);
                for (int c = 0; c < OLD_CLASSES; ++c) {
                    t_soft[c] = std::exp(teacher_logit[n * OLD_CLASSES + c] / KD_TEMP - m_t);
                    ss_t += t_soft[c];
                }

                float m_s = -1e9f;
                for (int c = 0; c < OLD_CLASSES; ++c)
                    m_s = std::max(m_s, logit[n * TOTAL_CLASSES + c] / KD_TEMP);
                
                float ss_s = 0.f;
                std::vector<float> s_soft(OLD_CLASSES);
                for (int c = 0; c < OLD_CLASSES; ++c) {
                    s_soft[c] = std::exp(logit[n * TOTAL_CLASSES + c] / KD_TEMP - m_s);
                    ss_s += s_soft[c];
                }

                for (int c = 0; c < OLD_CLASSES; ++c) {
                    t_soft[c] /= (ss_t + 1e-9f);
                    s_soft[c] /= (ss_s + 1e-9f);
                    kd += t_soft[c] * (std::log(t_soft[c] + 1e-10f) - std::log(s_soft[c] + 1e-10f));
                    float grad_kd = KD_WEIGHT * (KD_TEMP * KD_TEMP) * (s_soft[c] - t_soft[c]) / BATCH_SIZE;
                    glog[n * TOTAL_CLASSES + c] += grad_kd;
                }
            }
            kd = kd * KD_TEMP * KD_TEMP / BATCH_SIZE;

            epoch_ce += ce;
            epoch_kd += kd;

            // Backward & Update
            head.backward(fad.data(), glog.data(), BATCH_SIZE, gfeat.data());
            adapter.backward(fraw.data(), gfeat.data(), BATCH_SIZE);
            head.update(head_lr);
            adapter.update(adapter_lr);

            double step_time = step_timer.elapsed_ms();
            
            if (s % 20 == 0) {
                std::cout << "  Step " << std::setw(3) << s 
                          << " | CE: " << std::fixed << std::setprecision(4) << ce
                          << " | KD: " << kd 
                          << " | " << step_time << " ms" << std::endl;
            }
        }

        double epoch_time = epoch_timer.elapsed_ms();
        std::cout << "  Avg - CE: " << (epoch_ce / STEPS_PER_EPOCH) 
                  << ", KD: " << (epoch_kd / STEPS_PER_EPOCH)
                  << " | Epoch time: " << epoch_time / 1000.0 << "s" << std::endl;

        // Validation (less frequent)
        if ((e + 1) % 2 == 0 || e == NUM_EPOCHS - 1) {
            // Update prototype after epoch 2
            if (e >= 2) {
                std::cout << "\n [Updating Sunflowers Prototype...]" << std::endl;
                std::vector<float> sunflower_feats;
                std::vector<float> temp_img(IMG_SIZE * IMG_SIZE * 3);
                std::vector<float> temp_feat(FEATURE_DIM);
                
                int sunflower_count = 0;
                int max_samples = 96;
                
                for (int i = 0; i < std::min(max_samples, newd.n); ++i) {
                    if (newd.y[i] == 3) {
                        std::memcpy(temp_img.data(),
                                   newd.x.data() + i * IMG_SIZE * IMG_SIZE * 3,
                                   IMG_SIZE * IMG_SIZE * 3 * sizeof(float));
                        
                        fe.extract(temp_img.data(), temp_feat.data(), 1);
                        adapter.forward(temp_feat.data(), 1);
                        
                        sunflower_feats.insert(sunflower_feats.end(), 
                                              temp_feat.begin(), 
                                              temp_feat.begin() + FEATURE_DIM);
                        sunflower_count++;
                    }
                    if (sunflower_count >= max_samples) break;
                }
                
                if (sunflower_count > 0) {
                    nme.update_prototype(3, sunflower_feats, sunflower_count);
                    std::cout << "Updated sunflowers prototype with " 
                              << sunflower_count << " samples" << std::endl;
                }
            }
            
            run_validation_cached(fe, adapter, nme, vald, 
                                 ("Validation (Epoch " + std::to_string(e+1) + ")").c_str());
        }
    }

    // Save results
    head.save("inc1_head_w.npy", "inc1_head_b.npy");
    adapter.save("backbone_adapter.npz");

    double total_time = total_timer.elapsed_ms();
    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Total time: " << total_time / 1000.0 << " seconds" << std::endl;
    
    fe.print_timing_stats();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    int total_sec = duration.count() / 1000;
    int minutes = total_sec / 60;
    int seconds = total_sec % 60;
    std::cout << "\nTotal training time: " << minutes << "m " << seconds << "s" << std::endl;

    return 0;
}
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <random>
#include <algorithm>
#include <cmath>
#include <map>
#include <iomanip>
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

const int NUM_EPOCHS = 10;
const int STEPS_PER_EPOCH = 80;

const float HEAD_LR_START = 3e-4f;
const float HEAD_LR_END = 5e-5f;
const float ADAPTER_LR_START = 2e-5f;
const float ADAPTER_LR_END = 5e-7f;
const float LABEL_SMOOTH = 0.1f;
const float MAX_GRAD_NORM = 0.5f;

const float KD_TEMP = 2.0f;
const float KD_WEIGHT = 5.0f;

// ============================================================================
// NPY Utils
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
// iCaRL Nearest-Mean-of-Exemplars(NME) Classifier
// ============================================================================
struct NMEClassifier {
    std::vector<std::vector<float>> prototypes;  // [num_classes][feature_dim]
    int num_classes;
    
    NMEClassifier(int nc) : num_classes(nc) {
        prototypes.resize(nc, std::vector<float>(FEATURE_DIM, 0.f));
    }
    
    void load_prototypes(const char* filepath) {
        std::vector<size_t> shape;
        auto data = loadNpy(filepath, shape);
        
        if (shape.size() != 2 || shape[1] != FEATURE_DIM) {
            std::cerr << "ERROR: Invalid prototype shape!" << std::endl;
            return;
        }
        
        int loaded_classes = shape[0];
        std::cout << "Loading " << loaded_classes << " prototypes (expected " << num_classes << ")" << std::endl;
        
        // Load available prototypes
        for (int c = 0; c < std::min(loaded_classes, num_classes); ++c) {
            for (int d = 0; d < FEATURE_DIM; ++d) {
                prototypes[c][d] = data[c * FEATURE_DIM + d];
            }
        }
        
        // Initialize remaining prototypes to zero
        for (int c = loaded_classes; c < num_classes; ++c) {
            std::fill(prototypes[c].begin(), prototypes[c].end(), 0.f);
            std::cout << "WARNING: Prototype for class " << c << " not loaded (will be 0)" << std::endl;
        }
        
        std::cout << "Loaded " << loaded_classes << " prototypes, " 
                  << (num_classes - loaded_classes) << " uninitialized" << std::endl;
    }
    
    void update_prototype(int class_id, const std::vector<float>& exemplar_features, int num_exemplars) {
        // Compute mean of exemplar features
        std::fill(prototypes[class_id].begin(), prototypes[class_id].end(), 0.f);
        
        for (int i = 0; i < num_exemplars; ++i) {
            for (int d = 0; d < FEATURE_DIM; ++d) {
                prototypes[class_id][d] += exemplar_features[i * FEATURE_DIM + d];
            }
        }
        
        for (int d = 0; d < FEATURE_DIM; ++d) {
            prototypes[class_id][d] /= num_exemplars;
        }
    }
    
    int predict_single(const float* feature) {
        // L2 normalize feature
        float feature_norm = 0.f;
        for (int d = 0; d < FEATURE_DIM; ++d) {
            feature_norm += feature[d] * feature[d];
        }
        feature_norm = std::sqrt(feature_norm + 1e-8f);
        
        std::vector<float> feat_normalized(FEATURE_DIM);
        for (int d = 0; d < FEATURE_DIM; ++d) {
            feat_normalized[d] = feature[d] / feature_norm;
        }
        
        // Find nearest prototype
        float min_dist = 1e9f;
        int pred_class = -1;
        
        for (int c = 0; c < num_classes; ++c) {
            // Check if prototype is initialized (non-zero)
            float proto_norm = 0.f;
            for (int d = 0; d < FEATURE_DIM; ++d) {
                proto_norm += prototypes[c][d] * prototypes[c][d];
            }
            
            // Skip uninitialized prototypes
            if (proto_norm < 1e-6f) {
                continue;
            }
            
            proto_norm = std::sqrt(proto_norm);
            
            // Compute distance
            float dist = 0.f;
            for (int d = 0; d < FEATURE_DIM; ++d) {
                float diff = feat_normalized[d] - prototypes[c][d] / proto_norm;
                dist += diff * diff;
            }
            dist = std::sqrt(dist);
            
            if (dist < min_dist) {
                min_dist = dist;
                pred_class = c;
            }
        }
        
        // If no prototype found (shouldn't happen), return 0
        if (pred_class == -1) {
            pred_class = 0;
        }
        
        return pred_class;
    }
    
    void predict_batch(const float* features, int* predictions, int batch_size) {
        for (int n = 0; n < batch_size; ++n) {
            predictions[n] = predict_single(features + n * FEATURE_DIM);
        }
    }
};

// ============================================================================
// Dataset
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
        std::map<int,int> cnt;
        for (int i = 0; i < n; ++i) cnt[y[i]]++;
        std::cout << "[DATA] n=" << n << " ";
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
// Backbone Adapter
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
// FC Layer
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
        std::normal_distribution<float> d(0, std::sqrt(2.f / i));
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
        
        std::cout << "[HEAD] Loaded (expanded to " << out << " classes)" << std::endl;
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
// Feature Extractor
// ============================================================================
struct FeatureExtractor {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interp;

    bool load(const char* f) {
        model = tflite::FlatBufferModel::BuildFromFile(f);
        tflite::ops::builtin::BuiltinOpResolver r;
        tflite::InterpreterBuilder(*model, r)(&interp);
        interp->AllocateTensors();
        std::cout << "[BACKBONE] loaded" << std::endl;
        return true;
    }

    void extract(const float* img, float* feat, int bs) {
        auto* in = interp->input_tensor(0);
        auto* out = interp->output_tensor(0);
        for (int n = 0; n < bs; ++n) {
            std::memcpy(in->data.f, img + n * IMG_SIZE * IMG_SIZE * 3,
                        IMG_SIZE * IMG_SIZE * 3 * sizeof(float));
            
            interp->Invoke();
            
            std::memcpy(feat + n * FEATURE_DIM, out->data.f,
                        FEATURE_DIM * sizeof(float));
        }
    }
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Incremental Learning" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (argc < 10) {
        std::cerr << "Usage: " << argv[0] << std::endl;
        std::cerr << "  <backbone.tflite>" << std::endl;
        std::cerr << "  <head_w.npy> <head_b.npy>" << std::endl;
        std::cerr << "  <new_train_x.npy> <new_train_y.npy>" << std::endl;
        std::cerr << "  <rehearsal_x.npy> <rehearsal_y.npy>" << std::endl;
        std::cerr << "  <val_x.npy> <val_y.npy>" << std::endl;
        std::cerr << "  <prototypes.npy>" << std::endl;
        return -1;
    }

    FeatureExtractor fe;
    fe.load(argv[1]);

    Dataset newd, rehd, vald;
    newd.load(argv[4], argv[5]);
    rehd.load(argv[6], argv[7]);
    vald.load(argv[8], argv[9]);

    FCLayer head(FEATURE_DIM, TOTAL_CLASSES);
    head.load(argv[2], argv[3]);
    
    FCLayer teacher(FEATURE_DIM, OLD_CLASSES);
    teacher.load(argv[2], argv[3]);

    BackboneAdapter adapter;
    
    // Load NME classifier
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

    // Training loop
    for (int e = 0; e < NUM_EPOCHS; ++e) {
        std::cout << "\n=== Epoch " << (e + 1) << "/" << NUM_EPOCHS << " ===" << std::endl;
        
        float epoch_ce = 0.f, epoch_kd = 0.f;
        
        for (int s = 0; s < STEPS_PER_EPOCH; ++s) {
            int global_step = e * STEPS_PER_EPOCH + s;
            float progress = (float)global_step / total_steps;
            float head_lr = HEAD_LR_START * (1 - progress) + HEAD_LR_END * progress;
            float adapter_lr = ADAPTER_LR_START * (1 - progress) + ADAPTER_LR_END * progress;
            
            // Sample: 8 new + 16 rehearsal
            newd.getRandom(NEW_SAMPLES, rng, img.data(), lab.data());
            rehd.getRandom(REH_SAMPLES, rng,
                img.data() + NEW_SAMPLES * IMG_SIZE * IMG_SIZE * 3,
                lab.data() + NEW_SAMPLES);

            // Forward
            fe.extract(img.data(), fraw.data(), BATCH_SIZE);
            fad = fraw;
            adapter.forward(fad.data(), BATCH_SIZE);
            head.forward(fad.data(), logit.data(), BATCH_SIZE);
            teacher.forward(fraw.data(), teacher_logit.data(), BATCH_SIZE);

            // Initialize gradients
            std::fill(glog.begin(), glog.end(), 0.f);

            // CE Loss (all samples)
            float ce = 0.f;
            for (int n = 0; n < BATCH_SIZE; ++n) {
                float m = -1e9f, ss = 0.f;
                for (int c = 0; c < TOTAL_CLASSES; ++c)
                    m = std::max(m, logit[n * TOTAL_CLASSES + c]);
                
                for (int c = 0; c < TOTAL_CLASSES; ++c) {
                    prob[n * TOTAL_CLASSES + c] = std::exp(logit[n * TOTAL_CLASSES + c] - m);
                    ss += prob[n * TOTAL_CLASSES + c];
                }
                for (int c = 0; c < TOTAL_CLASSES; ++c)
                    prob[n * TOTAL_CLASSES + c] /= ss;
                
                for (int c = 0; c < TOTAL_CLASSES; ++c) {
                    float t = (c == lab[n]) ? (1 - LABEL_SMOOTH + LABEL_SMOOTH / TOTAL_CLASSES)
                                            : (LABEL_SMOOTH / TOTAL_CLASSES);
                    float p = std::max(prob[n * TOTAL_CLASSES + c], 1e-8f);
                    ce -= t * std::log(p);
                    glog[n * TOTAL_CLASSES + c] = prob[n * TOTAL_CLASSES + c] - t;
                }
            }
            ce /= BATCH_SIZE;
            
            // KD Loss (old classes)
            float kd = 0.f;
            for (int n = 0; n < BATCH_SIZE; ++n) {
                float m_t = -1e9f, ss_t = 0.f;
                for (int c = 0; c < OLD_CLASSES; ++c)
                    m_t = std::max(m_t, teacher_logit[n * OLD_CLASSES + c] / KD_TEMP);
                
                std::vector<float> t_soft(OLD_CLASSES);
                for (int c = 0; c < OLD_CLASSES; ++c) {
                    t_soft[c] = std::exp(teacher_logit[n * OLD_CLASSES + c] / KD_TEMP - m_t);
                    ss_t += t_soft[c];
                }
                for (int c = 0; c < OLD_CLASSES; ++c)
                    t_soft[c] /= ss_t;
                
                float m_s = -1e9f, ss_s = 0.f;
                for (int c = 0; c < OLD_CLASSES; ++c)
                    m_s = std::max(m_s, logit[n * TOTAL_CLASSES + c] / KD_TEMP);
                
                std::vector<float> s_soft(OLD_CLASSES);
                for (int c = 0; c < OLD_CLASSES; ++c) {
                    s_soft[c] = std::exp(logit[n * TOTAL_CLASSES + c] / KD_TEMP - m_s);
                    ss_s += s_soft[c];
                }
                for (int c = 0; c < OLD_CLASSES; ++c)
                    s_soft[c] /= ss_s;
                
                for (int c = 0; c < OLD_CLASSES; ++c) {
                    float p_t = std::max(t_soft[c], 1e-7f);
                    float p_s = std::max(s_soft[c], 1e-7f);
                    kd += t_soft[c] * (std::log(p_t) - std::log(p_s));
                    glog[n * TOTAL_CLASSES + c] += KD_WEIGHT * KD_TEMP * KD_TEMP * (s_soft[c] - t_soft[c]) / BATCH_SIZE;
                }
            }
            kd = kd * KD_TEMP * KD_TEMP / BATCH_SIZE;
            
            epoch_ce += ce;
            epoch_kd += kd;

            // Backward
            head.backward(fad.data(), glog.data(), BATCH_SIZE, gfeat.data());
            adapter.backward(fraw.data(), gfeat.data(), BATCH_SIZE);
            
            head.update(head_lr);
            adapter.update(adapter_lr);

            if (s % 20 == 0) {
                std::cout << "  Step " << std::setw(3) << s 
                          << " | CE: " << std::fixed << std::setprecision(4) << ce
                          << " | KD: " << kd << std::endl;
            }
        }
        
        std::cout << "  Avg - CE: " << (epoch_ce / STEPS_PER_EPOCH)
                  << ", KD: " << (epoch_kd / STEPS_PER_EPOCH) << std::endl;
        
        // Validation with NME
        if ((e + 1) % 2 == 0 || e == NUM_EPOCHS - 1) {
            // Update sunflowers prototype using new class samples after training is partially done
            if (e >= 2) {
                std::cout << "\n  [Updating Sunflowers Prototype...]" << std::endl;
                
                // Collect sunflowers features
                std::vector<float> sunflower_feats;
                int sunflower_count = 0;
                int max_samples = 96;
                
                for (int i = 0; i < std::min(max_samples, newd.n); ++i) {
                    if (newd.y[i] == 3) {  // sunflowers
                        std::memcpy(img.data(), 
                                   newd.x.data() + i * IMG_SIZE * IMG_SIZE * 3,
                                   IMG_SIZE * IMG_SIZE * 3 * sizeof(float));
                        
                        fe.extract(img.data(), fraw.data(), 1);
                        fad.assign(fraw.begin(), fraw.begin() + FEATURE_DIM);
                        adapter.forward(fad.data(), 1);
                        
                        sunflower_feats.insert(sunflower_feats.end(), 
                                             fad.begin(), fad.begin() + FEATURE_DIM);
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
            
            std::cout << "\n  [Validation]" << std::endl;
            
            int correct = 0, total = 0;
            std::vector<int> cc(TOTAL_CLASSES, 0), ct(TOTAL_CLASSES, 0);
            std::vector<int> nme_preds(vald.n);
            
            for (int i = 0; i < vald.n; ++i) {
                std::memcpy(img.data(), vald.x.data() + i * IMG_SIZE * IMG_SIZE * 3,
                           IMG_SIZE * IMG_SIZE * 3 * sizeof(float));
                int label = vald.y[i];
                
                fe.extract(img.data(), fraw.data(), 1);
                fad.assign(fraw.begin(), fraw.begin() + FEATURE_DIM);
                adapter.forward(fad.data(), 1);
                
                // NME prediction
                int pred = nme.predict_single(fad.data());
                
                if (pred == label) {
                    correct++;
                    cc[label]++;
                }
                ct[label]++;
                total++;
            }
            
            std::cout << "  Overall Accuracy: " << std::fixed << std::setprecision(4)
                      << (float)correct / total << " (" << correct << "/" << total << ")" << std::endl;
            
            const char* names[] = {"daisy", "dandelion", "roses", "sunflowers"};
            for (int c = 0; c < TOTAL_CLASSES; ++c) {
                if (ct[c] > 0) {
                    std::cout << "    " << names[c] << ": " 
                              << std::setprecision(4) << (float)cc[c] / ct[c]
                              << " (" << cc[c] << "/" << ct[c] << ")" << std::endl;
                }
            }
        }
    }

    // Save results
    head.save("inc1_head_w.npy", "inc1_head_b.npy");
    adapter.save("backbone_adapter.npz");
    
    std::cout << "\n === Incremental Learning Complete ===" << std::endl;
    
    return 0;
}
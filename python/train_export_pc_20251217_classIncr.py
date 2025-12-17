# -*- coding: utf-8 -*-
"""
Pretraining includes:

1. Herding exemplar selection
2. NME (Nearest-Mean-of-Exemplars) classification
3. Knowledge Distillation

These are came from iCaRL: Incremental Classifier and Representation Learning(2017)
"""

import os, time, tarfile, shutil, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# =============================================================================
# Paths
# =============================================================================
def pick_root():
    for root in (r"D:\\", r"C:\\"):
        if os.path.exists(root):
            return root
    return os.getcwd()

ROOT = pick_root()
BASE_DIR = r"D:\2025-1\lge\LGE_IncrmentalLearning\python"

WORK_DIR   = os.path.join(BASE_DIR, "work_icarl")
DATA_DIR   = os.path.join(BASE_DIR, "data_icarl")
CACHE_DIR  = os.path.join(WORK_DIR, "cache")
CKPT_DIR   = os.path.join(WORK_DIR, "ckpt")
EXPORT_DIR = os.path.join(WORK_DIR, "export")

for d in [WORK_DIR, DATA_DIR, CACHE_DIR, CKPT_DIR, EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# Hyperparameters
# =============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 7

BASE_CLASSES = ['daisy', 'dandelion', 'roses']   # Phase-1
NEW_CLASSES  = ['sunflowers']                   # Phase-2
CLASS_ORDER  = BASE_CLASSES + NEW_CLASSES

VAL_SPLIT = 0.2

NUM_EPOCHS_PHASE1 = 5
NUM_EPOCHS_PHASE2 = 8
STEPS_PER_EPOCH_PHASE2 = 80

HEAD_LR = 1e-3
BASE_LR = 1e-5
FINE_TUNE_FRACTION = 0.15

LABEL_SMOOTHING = 0.10
CLIP_NORM = 1.0

KD_T = 2.0
KD_LAMBDA = 2.0

EXEMPLARS_PER_CLASS = 96

PREFETCH_BUFFER = 1

np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
# Optimizer helper
# =============================================================================
def make_adam(lr: float):
    try:
        from tensorflow.keras.optimizers.legacy import Adam
        return Adam(learning_rate=lr)
    except:
        return tf.keras.optimizers.Adam(learning_rate=lr)

# =============================================================================
# Data utils
# =============================================================================
FLOWER_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

def _extract_flowers():
    print("\n[Data] flowers dataset download...")
    tgz_path = keras.utils.get_file(
        "flower_photos.tgz",
        origin=FLOWER_URL,
        cache_dir=DATA_DIR, cache_subdir="", extract=False
    )
    extract_root = Path(DATA_DIR) / "flower_photos_extracted"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    with tarfile.open(tgz_path, "r:gz") as tfp:
        tfp.extractall(path=extract_root)
    return extract_root / "flower_photos"

def _copy_subset(src_root: Path, dst_root: Path, class_names):
    if dst_root.exists():
        shutil.rmtree(dst_root)
    for cname in class_names:
        (dst_root / cname).mkdir(parents=True, exist_ok=True)
        imgs = sorted([p for p in (src_root / cname).glob("*") if p.is_file()])
        for img in imgs:
            shutil.copy(img, dst_root / cname)

def augment_raw(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.08)
    image = tf.image.random_contrast(image, 0.95, 1.05)
    return image

def preprocess(image, label, training=False):
    if training:
        image = augment_raw(image)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # [-1,1]
    return image, label

def make_ds_from_dir(root_dir, class_names, batch_size, training,
                     subset=None, validation_split=None,
                     use_cache=False, cache_tag=None):
    use_subset = (subset is not None)
    
    ds = keras.preprocessing.image_dataset_from_directory(
        directory=str(root_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=True if use_subset else training,
        seed=SEED,
        validation_split=(float(validation_split) if use_subset else None),
        subset=(subset if use_subset else None),
    )

    if use_cache:
        tag = cache_tag or f"{Path(root_dir).name}_{subset or 'train'}"
        ds = ds.cache(os.path.join(CACHE_DIR, f"{tag}.cache"))

    ds = ds.map(lambda x, y: preprocess(x, y, training=training),
                num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(1000, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.prefetch(PREFETCH_BUFFER)
    return ds

# =============================================================================
# iCaRL Exemplar Memory with Herding
# =============================================================================
class ExemplarMemory:
    """
    iCaRL Algorithm 4: Herding exemplar selection
    """
    def __init__(self, exemplars_per_class=16):
        self.m = exemplars_per_class
        self.images = None
        self.labels = None
        self.features = None  # ✓ Store features for NME
        self.class_to_indices = {}
        self.prototypes = None  # ✓ Class means for NME

    def __len__(self):
        return 0 if self.images is None else int(self.images.shape[0])

    def build_exemplars_herding(self, feature_extractor, ds_noaug, num_classes, class_ids):
        """
        Herding: Select exemplars that minimize distance to class mean
        """
        print("\nHerding exemplar selection...")

        # Extract all features
        xs, ys = [], []
        for x, y in ds_noaug:
            xs.append(x.numpy())
            ys.append(y.numpy())
        X = np.concatenate(xs, axis=0).astype(np.float32)
        Y = np.concatenate(ys, axis=0).astype(np.int32)

        print(f"  Total samples: {len(X)}")
        
        # Get features
        feats = feature_extractor.predict(X, batch_size=BATCH_SIZE, verbose=0)
        print(f"  Feature shape: {feats.shape}")

        keep_imgs, keep_lbls, keep_feats = [], [], []
        self.class_to_indices = {}

        for cid in class_ids:
            idxs = np.where(Y == cid)[0]
            if idxs.size == 0:
                continue

            # Class features
            feats_c = feats[idxs]  # [N_c, D]
            class_mean = np.mean(feats_c, axis=0, keepdims=True)  # [1, D]
            
            # iCaRL Herding Algorithm
            selected_indices = []
            selected_features = []
            
            for k in range(min(self.m, len(idxs))):
                # Running mean of selected exemplars
                if len(selected_features) == 0:
                    running_mean = np.zeros_like(class_mean)
                else:
                    running_mean = np.mean(selected_features, axis=0, keepdims=True)
                
                # Find exemplar that minimizes distance to class mean
                # φ = argmin || mean(selected + φ) - class_mean ||
                distances = []
                for i, feat_idx in enumerate(idxs):
                    if i in selected_indices:
                        distances.append(np.inf)
                        continue
                    
                    # What would the mean be if we add this exemplar?
                    candidate_mean = (running_mean * len(selected_features) + feats_c[i]) / (len(selected_features) + 1)
                    dist = np.linalg.norm(candidate_mean - class_mean)
                    distances.append(dist)
                
                best_local_idx = np.argmin(distances)
                selected_indices.append(best_local_idx)
                selected_features.append(feats_c[best_local_idx])

            # Store selected exemplars
            chosen_global = idxs[selected_indices]
            
            start_idx = sum(a.shape[0] for a in keep_imgs) if keep_imgs else 0
            self.class_to_indices[int(cid)] = list(range(start_idx, start_idx + len(chosen_global)))

            keep_imgs.append(X[chosen_global])
            keep_lbls.append(Y[chosen_global])
            keep_feats.append(feats[chosen_global])
            
            print(f"  Class {cid}: selected {len(chosen_global)}/{len(idxs)} exemplars")

        if keep_imgs:
            self.images = np.concatenate(keep_imgs, axis=0).astype(np.float32)
            self.labels = np.concatenate(keep_lbls, axis=0).astype(np.int32)
            self.features = np.concatenate(keep_feats, axis=0).astype(np.float32)
        else:
            self.images = np.zeros((0, IMG_SIZE, IMG_SIZE, 3), np.float32)
            self.labels = np.zeros((0,), np.int32)
            self.features = np.zeros((0, 1280), np.float32)

        # Compute prototypes (class means) for NME
        self.update_prototypes(num_classes)
        
        print(f"Total exemplars: {len(self)} samples")

    def update_prototypes(self, num_classes):
        """
        Compute class prototypes (means) for NME classification
        """
        self.prototypes = np.zeros((num_classes, self.features.shape[1]), dtype=np.float32)
        
        for c in range(num_classes):
            mask = (self.labels == c)
            if mask.sum() > 0:
                self.prototypes[c] = np.mean(self.features[mask], axis=0)
        
        print(f"Updated {num_classes} prototypes")

    def make_rehearsal_ds(self, batch_size):
        if self.images is None or len(self) == 0:
            return None
        ds = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        ds = ds.shuffle(min(len(self), 1000), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(PREFETCH_BUFFER)
        return ds

    def save(self, path):
        np.savez_compressed(
            path,
            m=self.m,
            images=self.images,
            labels=self.labels,
            features=self.features,
            prototypes=self.prototypes,
            class_to_indices=self.class_to_indices
        )

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        mem = ExemplarMemory(int(z["m"]))
        mem.images = z["images"]
        mem.labels = z["labels"]
        mem.features = z["features"]
        mem.prototypes = z["prototypes"]
        mem.class_to_indices = z["class_to_indices"].item()
        return mem

# =============================================================================
# NME Classifier
# =============================================================================
def nme_classify(features, prototypes):
    """
    iCaRL: Nearest-Mean-of-Exemplars classification
    
    Args:
        features: [N, D] query features
        prototypes: [C, D] class means
    
    Returns:
        predictions: [N] predicted class indices
    """
    # L2 normalize
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    prototypes_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    
    # Compute distances: [N, C]
    # dist[i, c] = ||features[i] - prototypes[c]||
    distances = np.zeros((features.shape[0], prototypes.shape[0]))
    for c in range(prototypes.shape[0]):
        diff = features_norm - prototypes_norm[c:c+1]
        distances[:, c] = np.linalg.norm(diff, axis=1)
    
    # Nearest prototype
    predictions = np.argmin(distances, axis=1)
    return predictions

# =============================================================================
# Model building
# =============================================================================
def build_base(backbone_fraction: float):
    base = MobileNetV2(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg',
        weights='imagenet'
    )
    
    # Freeze all
    for l in base.layers:
        l.trainable = False
    
    # Unfreeze last few conv layers
    if backbone_fraction > 0:
        conv_layers = [l for l in base.layers if isinstance(l, (layers.Conv2D, layers.DepthwiseConv2D))]
        k = max(1, int(len(conv_layers) * backbone_fraction))
        for l in conv_layers[-k:]:
            l.trainable = True
    
    return base

def build_head(num_classes):
    return keras.Sequential([
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(num_classes, kernel_regularizer=keras.regularizers.l2(1e-4))
    ], name="head")

def build_keras_models(num_classes, backbone_fraction):
    base = build_base(backbone_fraction)
    head = build_head(num_classes)
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    feat = base(inp, training=True)
    logits = head(feat, training=True)
    full = keras.Model(inp, logits)
    feat_model = keras.Model(inp, feat)
    return base, head, full, feat_model

# =============================================================================
# Evaluation with NME
# =============================================================================
def evaluate_nme(feat_extractor, ds, class_names, prototypes, label_offset=0):
    """
    Evaluate using NME (Nearest-Mean-of-Exemplars)
    """
    correct, total = 0, 0
    per_correct = np.zeros((len(class_names),), np.int64)
    per_total = np.zeros((len(class_names),), np.int64)

    for x, y_int in ds:
        # Extract features
        feats = feat_extractor(x, training=False).numpy()
        
        # NME classify
        pred = nme_classify(feats, prototypes)
        y = (y_int.numpy().astype(np.int32) + int(label_offset))

        correct += int(np.sum(pred == y))
        total += int(len(y))

        y_local = y_int.numpy().astype(np.int32)
        for c in range(len(class_names)):
            m = (y_local == c)
            per_total[c] += int(np.sum(m))
            per_correct[c] += int(np.sum((pred == (c + label_offset)) & m))

    acc = correct / max(1, total)
    per_acc = per_correct / np.maximum(1, per_total)
    return acc, per_acc, per_total

# =============================================================================
# Phase-1
# =============================================================================
def train_phase1(src_root: Path, subset_phase1: Path):
    num_classes = len(BASE_CLASSES)
    print(f"\n[Phase-1] iCaRL training: C={num_classes} {BASE_CLASSES}")

    _copy_subset(src_root, subset_phase1, BASE_CLASSES)

    train_ds = make_ds_from_dir(
        subset_phase1, BASE_CLASSES, BATCH_SIZE,
        training=True, subset="training", validation_split=VAL_SPLIT,
        use_cache=False
    )
    val_ds = make_ds_from_dir(
        subset_phase1, BASE_CLASSES, BATCH_SIZE,
        training=False, subset="validation", validation_split=VAL_SPLIT,
        use_cache=True, cache_tag="phase1_val"
    )

    base, head, full, feat_model = build_keras_models(num_classes, backbone_fraction=0.0)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Train
    full.compile(optimizer=make_adam(HEAD_LR), loss=loss_fn, metrics=["accuracy"])
    full.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS_PHASE1, verbose=1)

    # Build exemplar memory with Herding
    mem = ExemplarMemory(exemplars_per_class=EXEMPLARS_PER_CLASS)
    mem_build_ds = make_ds_from_dir(
        subset_phase1, BASE_CLASSES, BATCH_SIZE,
        training=False, subset="training", validation_split=VAL_SPLIT,
        use_cache=False
    )
    mem.build_exemplars_herding(feat_model, mem_build_ds, num_classes=num_classes,
                                class_ids=list(range(num_classes)))

    # Evaluate with NME
    print("\n[Phase-1] Evaluating with NME...")
    acc, per_acc, per_n = evaluate_nme(feat_model, val_ds, BASE_CLASSES, mem.prototypes, label_offset=0)
    print(f"NME Accuracy: {acc:.4f}")
    for i, c in enumerate(BASE_CLASSES):
        print(f"    {c:12s}: acc={per_acc[i]:.4f} | n={int(per_n[i])}")

    # Save
    w_path = os.path.join(CKPT_DIR, "phase1.weights.h5")
    full.save_weights(w_path)

    mem_path = os.path.join(CKPT_DIR, "memory_phase1.npz")
    mem.save(mem_path)

    print(f"\n[Phase-1] Complete!")
    print(f"  Weights: {w_path}")
    print(f"  Memory: {mem_path}")
    print(f"  Exemplars: {len(mem)}")
    print(f"  Prototypes shape: {mem.prototypes.shape}")

    # Export for C++
    export_for_cpp(feat_model, head, mem, num_classes)

    return w_path, mem_path

def export_for_cpp(feat_model, head, mem, num_classes):
    """
    Export data for C++ incremental learning
    """
    print("\n[Export] Preparing C++ files...")
    
    # 1. Backbone TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(feat_model)
    tflite_model = converter.convert()
    with open(os.path.join(EXPORT_DIR, "backbone_frozen.tflite"), "wb") as f:
        f.write(tflite_model)
    print("  ✓ backbone_frozen.tflite")

    # 2. Head weights
    head_dense = head.layers[-1]
    W, b = head_dense.get_weights()
    np.save(os.path.join(EXPORT_DIR, "phase1_head_w.npy"), W)
    np.save(os.path.join(EXPORT_DIR, "phase1_head_b.npy"), b)
    print(f"  ✓ head weights: {W.shape}")

    # 3. Exemplar data (for rehearsal)
    np.save(os.path.join(EXPORT_DIR, "rehearsal_images.npy"), mem.images)
    np.save(os.path.join(EXPORT_DIR, "rehearsal_labels.npy"), mem.labels)
    np.save(os.path.join(EXPORT_DIR, "rehearsal_features.npy"), mem.features)
    print(f"  ✓ rehearsal data: {mem.images.shape}")

    # 4. Prototypes for NME
    np.save(os.path.join(EXPORT_DIR, "prototypes.npy"), mem.prototypes)
    print(f"  ✓ prototypes: {mem.prototypes.shape}")
    
    # ============================================================================
    # 5. NEW CLASS Training Data (Phase-2에서 학습할 데이터)
    # ============================================================================
    print("\n[Export] Preparing NEW class data...")
    
    # NEW_CLASSES 데이터 준비
    src_root = Path(DATA_DIR) / "flower_photos_extracted" / "flower_photos"
    subset_new = Path(DATA_DIR) / "flow_subset_new"
    _copy_subset(src_root, subset_new, NEW_CLASSES)
    
    # Training split만 (validation은 나중에 합쳐서)
    new_train_ds = make_ds_from_dir(
        subset_new, NEW_CLASSES,
        batch_size=1, training=False,
        subset="training", validation_split=VAL_SPLIT
    )
    
    new_imgs, new_lbls = [], []
    for img, lbl in new_train_ds:
        new_imgs.append(img.numpy())
        # Global label: NEW_CLASSES는 3번 클래스부터 시작
        new_lbls.append((lbl.numpy() + num_classes).astype(np.int32))
    
    new_imgs = np.concatenate(new_imgs, axis=0)
    new_lbls = np.concatenate(new_lbls, axis=0)
    
    np.save(os.path.join(EXPORT_DIR, "new_train_images.npy"), new_imgs)
    np.save(os.path.join(EXPORT_DIR, "new_train_labels.npy"), new_lbls)
    print(f"  ✓ new_train_images.npy: {new_imgs.shape}")
    print(f"  ✓ new_train_labels.npy: {new_lbls.shape}")
    print(f"    Label range: [{new_lbls.min()}, {new_lbls.max()}]")
    
    # ============================================================================
    # 6. VALIDATION Data (모든 클래스 포함)
    # ============================================================================
    print("\n[Export] Preparing VALIDATION data (all classes)...")
    
    # Old classes validation
    subset_old = Path(DATA_DIR) / "flow_subset_phase1_base"
    old_val_ds = make_ds_from_dir(
        subset_old, BASE_CLASSES,
        batch_size=1, training=False,
        subset="validation", validation_split=VAL_SPLIT
    )
    
    old_val_imgs, old_val_lbls = [], []
    for img, lbl in old_val_ds:
        old_val_imgs.append(img.numpy())
        old_val_lbls.append(lbl.numpy().astype(np.int32))
    
    old_val_imgs = np.concatenate(old_val_imgs, axis=0)
    old_val_lbls = np.concatenate(old_val_lbls, axis=0)
    
    # New classes validation
    new_val_ds = make_ds_from_dir(
        subset_new, NEW_CLASSES,
        batch_size=1, training=False,
        subset="validation", validation_split=VAL_SPLIT
    )
    
    new_val_imgs, new_val_lbls = [], []
    for img, lbl in new_val_ds:
        new_val_imgs.append(img.numpy())
        new_val_lbls.append((lbl.numpy() + num_classes).astype(np.int32))
    
    new_val_imgs = np.concatenate(new_val_imgs, axis=0)
    new_val_lbls = np.concatenate(new_val_lbls, axis=0)
    
    val_all_imgs = np.concatenate([old_val_imgs, new_val_imgs], axis=0)
    val_all_lbls = np.concatenate([old_val_lbls, new_val_lbls], axis=0)
    
    indices = np.random.RandomState(SEED).permutation(len(val_all_lbls))
    val_all_imgs = val_all_imgs[indices]
    val_all_lbls = val_all_lbls[indices]
    
    np.save(os.path.join(EXPORT_DIR, "val_all_images.npy"), val_all_imgs)
    np.save(os.path.join(EXPORT_DIR, "val_all_labels.npy"), val_all_lbls)
    print(f"  ✓ val_all_images.npy: {val_all_imgs.shape}")
    print(f"  ✓ val_all_labels.npy: {val_all_lbls.shape}")
    
    unique, counts = np.unique(val_all_lbls, return_counts=True)
    print(f"  Label distribution:")
    for c, cnt in zip(unique, counts):
        print(f"    Class {c}: {cnt} samples")

    print(f"\n[Export] Complete! Files in {EXPORT_DIR}")
    print("=" * 60)
    print("Required files for C++:")
    print("  1. backbone_frozen.tflite")
    print("  2. phase1_head_w.npy")
    print("  3. phase1_head_b.npy")
    print("  4. new_train_images.npy")
    print("  5. new_train_labels.npy")
    print("  6. rehearsal_images.npy")
    print("  7. rehearsal_labels.npy")
    print("  8. val_all_images.npy")
    print("  9. val_all_labels.npy")
    print(" 10. prototypes.npy")
    print("=" * 60)

# =============================================================================
# Main
# =============================================================================
def main():
    src_root = _extract_flowers()
    subset_phase1 = Path(DATA_DIR) / "flow_subset_phase1_base"
    
    w0, mem0 = train_phase1(src_root, subset_phase1)
    
    print("\n=== Phase-1(Pretraining) Complete ===")

if __name__ == "__main__":
    main()
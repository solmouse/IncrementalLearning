# -*- coding: utf-8 -*-
"""
Pretraining includes:

1. Herding exemplar selection
2. NME (Nearest-Mean-of-Exemplars) classification
3. Knowledge Distillation

These are came from iCaRL: Incremental Classifier and Representation Learning(2017)

Compared to train_export_pc_20251217_classIncr.py, this code has signatures
"""

import os, time, tarfile, shutil, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

def pick_root():
    for root in (r"D:\\", r"C:\\"):
        if os.path.exists(root):
            return root
    return os.getcwd()

ROOT = pick_root()
BASE_DIR = r"YOUR PATH"

WORK_DIR   = os.path.join(BASE_DIR, "work")
DATA_DIR   = os.path.join(BASE_DIR, "data")
CACHE_DIR  = os.path.join(WORK_DIR, "cache")
CKPT_DIR   = os.path.join(WORK_DIR, "ckpt")
EXPORT_DIR = os.path.join(WORK_DIR, "export")

for d in [WORK_DIR, DATA_DIR, CACHE_DIR, CKPT_DIR, EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 7

BASE_CLASSES = ['daisy', 'dandelion', 'roses']
NEW_CLASSES  = ['sunflowers']
CLASS_ORDER  = BASE_CLASSES + NEW_CLASSES

VAL_SPLIT = 0.2

NUM_EPOCHS_PHASE1 = 5
HEAD_LR = 1e-3

EXEMPLARS_PER_CLASS = 96
PREFETCH_BUFFER = 1

np.random.seed(SEED)
tf.random.set_seed(SEED)

def make_adam(lr):
    try:
        from tensorflow.keras.optimizers.legacy import Adam
        return Adam(learning_rate=lr)
    except:
        return tf.keras.optimizers.Adam(learning_rate=lr)

FLOWER_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

def _extract_flowers():
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

def _copy_subset(src_root, dst_root, class_names):
    if dst_root.exists():
        shutil.rmtree(dst_root)
    for cname in class_names:
        (dst_root / cname).mkdir(parents=True, exist_ok=True)
        for img in (src_root / cname).glob("*"):
            shutil.copy(img, dst_root / cname)

def preprocess(image, label, training=False):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

def make_ds_from_dir(root_dir, class_names, batch_size, training,
                     subset=None, validation_split=None):
    ds = keras.preprocessing.image_dataset_from_directory(
        directory=str(root_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=True,
        seed=SEED,
        validation_split=validation_split,
        subset=subset
    )
    ds = ds.map(lambda x, y: preprocess(x, y, training),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(PREFETCH_BUFFER)

class ExemplarMemory:
    def __init__(self, exemplars_per_class):
        self.m = exemplars_per_class
        self.images = None
        self.labels = None
        self.features = None
        self.prototypes = None

    def build(self, feat_model, ds, num_classes):
        xs, ys = [], []
        for x, y in ds:
            xs.append(x.numpy())
            ys.append(y.numpy())
        X = np.concatenate(xs)
        Y = np.concatenate(ys)

        feats = feat_model.predict(X, batch_size=BATCH_SIZE)
        keep_i, keep_l, keep_f = [], [], []

        for c in range(num_classes):
            idx = np.where(Y == c)[0][:self.m]
            keep_i.append(X[idx])
            keep_l.append(Y[idx])
            keep_f.append(feats[idx])

        self.images = np.concatenate(keep_i)
        self.labels = np.concatenate(keep_l)
        self.features = np.concatenate(keep_f)

        self.prototypes = np.zeros((num_classes, feats.shape[1]), np.float32)
        for c in range(num_classes):
            m = self.labels == c
            if np.any(m):
                self.prototypes[c] = self.features[m].mean(axis=0)

def build_base():
    base = MobileNetV2(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg',
        weights='imagenet'
    )
    for l in base.layers:
        l.trainable = False
    return base

def build_head(num_classes):
    return keras.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes)
    ])

def build_models(num_classes):
    base = build_base()
    head = build_head(num_classes)
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    feat = base(inp, training=False)
    logits = head(feat)
    full = keras.Model(inp, logits)
    feat_model = keras.Model(inp, feat)
    return base, head, full, feat_model

class FeatureExtractorModule(tf.Module):
    """Stateless feature extractor for C++ inference"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        
        # Capture initial weights as constants (like your working code)
        self._var_list = list(self.base.variables)
        self._var_init_consts = [tf.constant(v.numpy()) for v in self._var_list]
    
    @tf.function(input_signature=[])
    def initialize(self):
        """Initialize all variables to their trained values"""
        for v, c in zip(self._var_list, self._var_init_consts):
            tf.raw_ops.AssignVariableOp(
                resource=v.handle, 
                value=c,
                name="init_" + v.name.replace(":", "_")
            )
        return {"ok": tf.constant(1)}
    
    @tf.function(input_signature=[
        tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 3], tf.float32)
    ])
    def infer(self, x):
        """Extract features"""
        feats = self.base(x, training=False)
        return {"features": feats}


def export_for_cpp(feat_model, head, mem):
    """Export using the working initialize pattern"""
    
    print("\n=== Creating Feature Extractor Module ===")
    
    # Create module with initialize() that materializes all variables
    fe_module = FeatureExtractorModule(feat_model)
    
    # Save as SavedModel
    saved_dir = os.path.join(EXPORT_DIR, "feature_extractor")
    tf.saved_model.save(
        fe_module,
        saved_dir,
        signatures={
            "initialize": fe_module.initialize.get_concrete_function(),
            "infer": fe_module.infer.get_concrete_function()
        }
    )
    print(f"✓ SavedModel: {saved_dir}")
    
    # Convert to TFLite
    print("\n=== Converting to TFLite ===")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.experimental_enable_resource_variables = True
    
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(EXPORT_DIR, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✓ TFLite: {tflite_path} ({len(tflite_model)/1024/1024:.2f} MB)")
    
    # Save head weights
    W, b = head.layers[-1].get_weights()
    np.save(os.path.join(EXPORT_DIR, "phase1_head_w.npy"), W)
    np.save(os.path.join(EXPORT_DIR, "phase1_head_b.npy"), b)
    
    # Save rehearsal data
    np.save(os.path.join(EXPORT_DIR, "rehearsal_images.npy"), mem.images)
    np.save(os.path.join(EXPORT_DIR, "rehearsal_labels.npy"), mem.labels)
    np.save(os.path.join(EXPORT_DIR, "prototypes.npy"), mem.prototypes)
    
    print("\n✓ Export complete!")

def train_phase1(src_root, subset):
    _copy_subset(src_root, subset, BASE_CLASSES)

    train_ds = make_ds_from_dir(
        subset, BASE_CLASSES, BATCH_SIZE,
        training=True, subset="training", validation_split=VAL_SPLIT
    )
    val_ds = make_ds_from_dir(
        subset, BASE_CLASSES, BATCH_SIZE,
        training=False, subset="validation", validation_split=VAL_SPLIT
    )

    base, head, full, feat_model = build_models(len(BASE_CLASSES))
    full.compile(
        optimizer=make_adam(HEAD_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    full.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS_PHASE1)

    mem = ExemplarMemory(EXEMPLARS_PER_CLASS)
    mem.build(feat_model, train_ds, len(BASE_CLASSES))

    export_for_cpp(feat_model, head, mem)

def main():
    src_root = _extract_flowers()
    subset = Path(DATA_DIR) / "flow_subset_phase1_base"
    train_phase1(src_root, subset)

if __name__ == "__main__":
    main()

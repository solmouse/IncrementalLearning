# -*- coding: utf-8 -*-

import os, time, zipfile, shutil, random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# -------------------------------
# 0) ROOT
# -------------------------------

CKPT_BEFORE     = 'ckpt_before/checkpoint'
CKPT_AFTER      = 'ckpt_after/ckpt_updated'
TFLITE_PATH     = 'ckpt_before/model.tflite'

# -------------------------------
# 1) HyperParam
# -------------------------------
IMG_SIZE = 192
NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_TRAIN_SAMPLES = 500
NUM_EPOCHS = 5
EXEMPLARS_PER_CLASS = 10
LABEL_SMOOTHING = 0.05
LR = 3e-4
CLIP_NORM = 1.0

# -------------------------------
# 2) Data (fake data for incremental testing)
# -------------------------------
def augment_raw(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image

def preprocess(image, label, training=False):
    if training:
        image = augment_raw(image)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

SYN_TRAIN = NUM_TRAIN_SAMPLES
SYN_VAL = 200

def make_raw_ds(n, shuffle=True):
    x = tf.random.uniform([n, IMG_SIZE, IMG_SIZE, 3], 0, 255, dtype=tf.float32)
    y = tf.random.uniform([n], minval=0, maxval=NUM_CLASSES, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.batch(BATCH_SIZE)
    return ds

_raw_train = make_raw_ds(SYN_TRAIN, True)
_raw_val = make_raw_ds(SYN_VAL, False)

train_ds = (_raw_train
    .map(lambda x,y: preprocess(x, tf.cast(y, tf.int32), training=True),
        num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE))

test_ds = (_raw_val
    .map(lambda x,y: preprocess(x, tf.cast(y, tf.int32), training=False),
        num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE))

# -------------------------------
# 3) Exemplar (herding)
# -------------------------------
class ExemplarManager:
    def __init__(self, base_model, capacity_per_class=10):
        self.base = base_model
        self.capacity = capacity_per_class
        self.bank = {c: [] for c in range(NUM_CLASSES)}
    @tf.function
    def _featurize(self, x):
        return self.base(x, training=False)
    def add_candidates(self, images, labels):
        if self.capacity <= 0: return
        feats = self._featurize(images)
        feats_np = feats.numpy(); labels_np = labels.numpy(); images_np = images.numpy()
        for i in range(len(images_np)):
            c = int(np.argmax(labels_np[i]))
            self.bank[c].append((images_np[i], labels_np[i], feats_np[i]))
        for c in range(NUM_CLASSES):
            items = self.bank[c]
            if len(items) <= self.capacity: continue
            feats_c = np.stack([f for (_,_,f) in items], axis=0)
            mean_c = feats_c.mean(axis=0, keepdims=True)
            d = np.linalg.norm(feats_c - mean_c, axis=1)
            idx = np.argsort(d)[: self.capacity]
            self.bank[c] = [items[j] for j in idx]
    def make_dataset(self, batch_size=BATCH_SIZE):
        if self.capacity <= 0: return None
        imgs, lbs = [], []
        for c in range(NUM_CLASSES):
            for (im, lb, _) in self.bank[c]:
                imgs.append(im); lbs.append(lb)
        if not imgs: return None
        x = tf.convert_to_tensor(np.stack(imgs, 0), dtype=tf.float32)
        y = tf.convert_to_tensor(np.stack(lbs, 0), dtype=tf.float32)
        return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(512).batch(batch_size).repeat()

# -------------------------------
# 4) BiC-lite
# -------------------------------
class BiCLite:
    def __init__(self):
        self.a = 1.0
        self.b = np.zeros((NUM_CLASSES,), dtype=np.float32)
    def fit(self, logits, labels, eps=1e-4):
        n, k = logits.shape
        smoothed = (1.0 - LABEL_SMOOTHING) * labels + LABEL_SMOOTHING / (k - 1)
        target = np.log(np.clip(smoothed, eps, 1 - eps)) - np.log(np.clip(1 - smoothed, eps, 1 - eps))
        A = logits
        a = float(np.sum(A*target) / (np.sum(A*A) + 1e-8))
        b = target.mean(axis=0) - a * A.mean(axis=0)
        self.a = a; self.b = b.astype(np.float32)
    def apply(self, logits):
        return self.a * logits + self.b


# -------------------------------
# 5) Incremental training
# -------------------------------
print("\n" + "="*60)
print(">>> Incr training test")
print("="*60)

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH, experimental_delegates=[])
interpreter.allocate_tensors()
init_fn    = interpreter.get_signature_runner("initialize"); init_fn()
train_fn   = interpreter.get_signature_runner("train")
infer_fn   = interpreter.get_signature_runner("infer")
save_fn    = interpreter.get_signature_runner("save")
restore_fn = interpreter.get_signature_runner("restore")

print("\n Restore checkpt:", CKPT_BEFORE)
restore_fn(flat_weights=np.load(CKPT_BEFORE + '.npy').astype(np.float32))
print(" Restore done")

rehearsal_ds = None
initial_data_for_exemplars = train_ds.take(10)
incremental_update_data    = train_ds.skip(10).take(10)
if EXEMPLARS_PER_CLASS > 0:
    base = MobileNetV2(include_top=False, pooling='avg', weights='imagenet'); base.trainable=False
    ex_mgr = ExemplarManager(base, capacity_per_class=EXEMPLARS_PER_CLASS)
    for bx, by in initial_data_for_exemplars:
        ex_mgr.add_candidates(bx, by)
    rehearsal_ds = ex_mgr.make_dataset(batch_size=BATCH_SIZE)

print("\n Incremental training start")
steps = 0
if rehearsal_ds:
    update_ds = tf.data.Dataset.zip((incremental_update_data, rehearsal_ds))
    for (bx, by), (exx, exy) in update_ds:
        out_new = train_fn(x=np.array(bx, dtype=np.float32), y=np.array(by, dtype=np.float32))
        print(f"  ✓ Step {steps+1} (new)  Loss: {float(out_new['loss']):.4f}"); steps += 1
        out_rep = train_fn(x=np.array(exx, dtype=np.float32), y=np.array(exy, dtype=np.float32))
        print(f"  ✓ Step {steps+1} (repr) Loss: {float(out_rep['loss']):.4f}"); steps += 1
else:
    for bx, by in incremental_update_data:
        out = train_fn(x=np.array(bx, dtype=np.float32), y=np.array(by, dtype=np.float32))
        print(f"  ✓ Step {steps+1} (new)  Loss: {float(out['loss']):.4f}"); steps += 1

print("\n Save updated ckpt:", CKPT_AFTER)
np.save(CKPT_AFTER, save_fn()['weights'])
print("Update done")


# -------------------------------
# 6) Evaluation
# -------------------------------
def evaluate_with_ckpt(tflite_path, ckpt_npy, dataset):
    itp = tf.lite.Interpreter(model_path=tflite_path)
    itp.allocate_tensors()
    itp.get_signature_runner("initialize")()
    restore_= itp.get_signature_runner("restore")
    infer_  = itp.get_signature_runner("infer")
    restore_(flat_weights=np.load(ckpt_npy).astype(np.float32))
    total=correct=0
    for x, y in dataset:
        out = infer_(x=np.array(x, dtype=np.float32))
        pred = np.argmax(out['output'], axis=1); lab = np.argmax(y.numpy(), axis=1)
        correct += np.sum(pred==lab); total += len(lab)
    return correct/max(total,1)

def collect_logits(interp, ckpt_npy, dataset, max_batches=5):
    interp.get_signature_runner("initialize")()
    restore_= interp.get_signature_runner("restore")
    infer_  = interp.get_signature_runner("infer")
    restore_(flat_weights=np.load(ckpt_npy).astype(np.float32))
    logits_list, labels_list = [], []
    for i,(x,y) in enumerate(dataset):
        if i>=max_batches: break
        out = infer_(x=np.array(x, dtype=np.float32))
        logits_list.append(out['logits']); labels_list.append(y.numpy())
    return np.concatenate(logits_list,0), np.concatenate(labels_list,0)

print("\n" + "="*60)
print("Eval and BiC-lite")
print("="*60)
acc_before = evaluate_with_ckpt(TFLITE_PATH, CKPT_BEFORE + '.npy', test_ds)
acc_after  = evaluate_with_ckpt(TFLITE_PATH, CKPT_AFTER + '.npy',  test_ds)
print(f"🔹 ACC before update : {acc_before:.2%}")
print(f"🔹 ACC after update : {acc_after:.2%}")

itp_tmp = tf.lite.Interpreter(model_path=TFLITE_PATH); itp_tmp.allocate_tensors()
logits_val, labels_val = collect_logits(itp_tmp, CKPT_AFTER + '.npy', test_ds, max_batches=5)
bic = BiCLite(); bic.fit(logits_val, labels_val)

itp_bic = tf.lite.Interpreter(model_path=TFLITE_PATH); itp_bic.allocate_tensors()
itp_bic.get_signature_runner("initialize")()
restore_b = itp_bic.get_signature_runner("restore"); infer_b = itp_bic.get_signature_runner("infer")
restore_b(flat_weights=np.load(CKPT_AFTER + '.npy').astype(np.float32))
correct=total=0
for x, y in test_ds:
    out = infer_b(x=np.array(x, dtype=np.float32))
    logits_adj = bic.apply(out['logits'])
    probs = tf.nn.softmax(logits_adj, axis=-1).numpy()
    pred = np.argmax(probs, axis=1); lab = np.argmax(y.numpy(), axis=1)
    correct += np.sum(pred==lab); total += len(lab)
print(f"ACC after BiC-lite adoption: {correct/max(total,1):.2%}")
print("="*60)

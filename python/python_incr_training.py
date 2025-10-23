# -*- coding: utf-8 -*-
"""
MobileNetV2 온디바이스 증분학습 (flowers 4 classes, Balanced 프리셋)
- 단일 드라이브 경로 통일
- initialize(): 모든 변수 materialize (READ_VARIABLE 방지)
- Phase-1: Python 학습 (Adam), 베이스 일부만 미세조정
- Phase-2: TFLite에서 헤드만 SGD로 증분학습 (+ 리허설)
"""

import os, time, tarfile, shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# -------------------------------
# 드라이브/경로
# -------------------------------
def pick_root():
    for root in (r"D:\\", r"C:\\"):
        if os.path.exists(root):
            return root
    return os.getcwd()

ROOT = pick_root()
WORK_DIR  = os.path.join(ROOT, "tf_runs", "mobilenetv2_incremental_flowers4_balanced")
DATA_DIR  = os.path.join(WORK_DIR, "data")
CACHE_DIR = os.path.join(WORK_DIR, "cache")
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["KERAS_HOME"] = os.path.join(WORK_DIR, "keras_home")
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)

SAVED_MODEL_DIR = os.path.join(WORK_DIR, "saved_model")
CKPT_BEFORE     = os.path.join(WORK_DIR, "checkpoint")
CKPT_AFTER      = os.path.join(WORK_DIR, "checkpoint_updated")
TFLITE_PATH     = os.path.join(WORK_DIR, "model.tflite")

# -------------------------------
# 하이퍼파라미터 (Balanced 프리셋)
# -------------------------------
IMG_SIZE = 224
NUM_CLASSES = 4
BATCH_SIZE = 32

# 데모/속도 위해 일부만 사용하려면 숫자 지정, 전량 사용하려면 None
NUM_TRAIN_SAMPLES = None         # None이면 전량 사용
NUM_EPOCHS = 3

# 정규화/안정화
LABEL_SMOOTHING = 0.10
CLIP_NORM = 1.0

# Phase-1 (Python) 학습
HEAD_LR = 8e-4
BASE_LR = HEAD_LR * 0.1
FINE_TUNE_FRACTION = 0.25        # 베이스 뒤쪽 25% conv 레이어만 미세조정

# 리허설
EXEMPLARS_PER_CLASS = 16

# Phase-2 (TFLite) 학습
LR_TFLITE_SGD = 3e-4
INCR_NEW_STEPS = 60
INCR_REHEARSAL_STEPS = 24

# -------------------------------
# 데이터 (flowers 4 classes)
# -------------------------------
FLOWER_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
SUBSET_CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers']

print("\n데이터셋 로딩 (flowers 4 classes)...")
tgz_path = keras.utils.get_file(
    "flower_photos.tgz",
    origin=FLOWER_URL,
    cache_dir=DATA_DIR, cache_subdir="", extract=False
)
extract_root = Path(DATA_DIR) / "flower_photos_extracted"
subset_root  = Path(DATA_DIR) / "flower_photos_subset4"
if extract_root.exists(): shutil.rmtree(extract_root)
if subset_root.exists():  shutil.rmtree(subset_root)
with tarfile.open(tgz_path, "r:gz") as tfp:
    tfp.extractall(path=extract_root)

src_root = extract_root / "flower_photos"
subset_root.mkdir(parents=True, exist_ok=True)
for cname in SUBSET_CLASSES:
    shutil.copytree(src_root / cname, subset_root / cname)

SEED = 42
_raw_train = keras.preprocessing.image_dataset_from_directory(
    subset_root, labels="inferred", label_mode="int",
    validation_split=0.2, subset="training", seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=True,
)

_raw_val = keras.preprocessing.image_dataset_from_directory(
    subset_root, labels="inferred", label_mode="int",
    validation_split=0.2, subset="validation", seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=False,
)

def augment_raw(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.08)
    image = tf.image.random_contrast(image, 0.95, 1.05)
    return image

def preprocess(image, label, training=False):
    if training:
        image = augment_raw(image)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # [-1,1]
    label = tf.one_hot(tf.cast(label, tf.int32), NUM_CLASSES)
    return image, label

train_raw = _raw_train
if NUM_TRAIN_SAMPLES:
    train_raw = _raw_train.take(max(1, NUM_TRAIN_SAMPLES // BATCH_SIZE))

train_ds = (train_raw
    .map(lambda x,y: preprocess(x, y, training=True),  num_parallel_calls=tf.data.AUTOTUNE)
    .cache(os.path.join(CACHE_DIR, "train.cache"))
    .prefetch(tf.data.AUTOTUNE))

test_ds = (_raw_val
    .map(lambda x,y: preprocess(x, y, training=False), num_parallel_calls=tf.data.AUTOTUNE)
    .cache(os.path.join(CACHE_DIR, "val.cache"))
    .prefetch(tf.data.AUTOTUNE))

# -------------------------------
# Exemplar manager (herding)
# -------------------------------
class ExemplarManager:
    def __init__(self, base_model, capacity_per_class=16):
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
# 모델 (initialize 패치 + Phase-1/2)
# -------------------------------
class IncrementalModel(tf.Module):
    def __init__(self, num_classes=NUM_CLASSES, fine_tune_fraction=0.25):
        super().__init__()
        self.base = MobileNetV2(include_top=False, pooling='avg', weights='imagenet')

        # 전체 freeze 후 뒤쪽 일부 conv만 unfreeze
        for l in self.base.layers:
            l.trainable = False
        conv_layers = [l for l in self.base.layers
                       if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))]
        k_unfreeze = max(1, int(len(conv_layers) * fine_tune_fraction))
        for l in conv_layers[-k_unfreeze:]:
            l.trainable = True

        # 헤드 (L2 regularization 추가)
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', name='fc1',
                                  kernel_regularizer=regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu', name='fc2',
                                  kernel_regularizer=regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, name='logits',
                                  kernel_regularizer=regularizers.l2(1e-4)),
        ])

        # build variables
        dummy = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3], tf.float32)
        _ = self.head(self.base(dummy, training=False), training=False)

        # Phase-1 optimizers
        self.opt_head = tf.keras.optimizers.Adam(learning_rate=HEAD_LR)
        self.opt_base = tf.keras.optimizers.Adam(learning_rate=BASE_LR)

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                               label_smoothing=LABEL_SMOOTHING)
        self.lr_sgd = tf.Variable(LR_TFLITE_SGD, dtype=tf.float32, trainable=False)

        # initialize()에 쓸 스냅샷
        self._var_list = list(self.base.variables) + list(self.head.variables)
        self._var_init_consts = [tf.constant(v.numpy()) for v in self._var_list]

    # -------- Phase-1: Python --------
    @tf.function
    def train_step_python(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            feats  = self.base(x, training=False)
            logits = self.head(feats, training=True)
            loss   = tf.reduce_mean(self.loss_fn(y, logits))
        # head
        h_vars  = self.head.trainable_variables
        h_grads = tape.gradient(loss, h_vars)
        h_grads = [tf.clip_by_norm(g, CLIP_NORM) if g is not None else None for g in h_grads]
        self.opt_head.apply_gradients([(g,w) for g,w in zip(h_grads, h_vars) if g is not None])
        # base
        b_vars = [v for v in self.base.trainable_variables if v.trainable]
        if b_vars:
            b_grads = tape.gradient(loss, b_vars)
            b_grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in b_grads]
            self.opt_base.apply_gradients([(g,w) for g,w in zip(b_grads, b_vars) if g is not None])
        del tape
        return loss

    # -------- Phase-2: TFLite --------
    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32),
        tf.TensorSpec([None, NUM_CLASSES], tf.float32)
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            feats = self.base(x, training=False)
            logits = self.head(feats, training=True)
            loss = tf.reduce_mean(self.loss_fn(y, logits))
        grads = tape.gradient(loss, self.head.trainable_variables)
        grads = [tf.clip_by_norm(g, CLIP_NORM) if g is not None else None for g in grads]
        for w, g in zip(self.head.trainable_variables, grads):
            if g is not None:
                w.assign_sub(self.lr_sgd * g)
        return {"loss": loss}

    @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)])
    def infer(self, x):
        feats = self.base(x, training=False)
        logits = self.head(feats, training=False)
        probs = tf.nn.softmax(logits, axis=-1)
        return {"output": probs, "logits": logits}

    @tf.function(input_signature=[])
    def save(self):
        flat = tf.concat([tf.reshape(w, [-1]) for w in self.head.trainable_variables], axis=0)
        return {"weights": flat}

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def restore(self, flat_weights):
        offset = 0
        for w in self.head.trainable_variables:
            size = tf.reduce_prod(w.shape)
            new_val = tf.reshape(flat_weights[offset:offset+size], w.shape)
            w.assign(new_val)
            offset += size
        return {"restored": True}

    @tf.function(input_signature=[])
    def initialize(self):
        for v, c in zip(self._var_list, self._var_init_consts):
            tf.raw_ops.AssignVariableOp(resource=v.handle, value=c, name="init_"+v.name.replace(":","_"))
        return {"ok": True}

# -------------------------------
# Phase-1: Python 기준선 학습
# -------------------------------
m = IncrementalModel(num_classes=NUM_CLASSES, fine_tune_fraction=FINE_TUNE_FRACTION)
for images, _ in train_ds.take(1):
    _ = m.infer(images)

print("\n[Phase-1] Python 학습 시작...")
for epoch in range(NUM_EPOCHS):
    losses = []
    tic = time.time()
    for step, (x, y) in enumerate(train_ds, 1):
        loss = m.train_step_python(x, y)
        losses.append(float(loss))
        if step % 20 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Step {step} - Loss {float(loss):.4f}")
    print(f"✅ Epoch {epoch+1} | 평균 Loss: {np.mean(losses):.4f} | {time.time()-tic:.1f}s")

def eval_python(model, dataset):
    total=correct=0
    for x,y in dataset:
        out = model.infer(x)
        pred = tf.argmax(out['output'], axis=1).numpy()
        lab  = tf.argmax(y, axis=1).numpy()
        correct += (pred==lab).sum(); total += len(lab)
    return correct / max(total,1)

print("\n📊 (파이썬) 기준선 정확도...")
py_acc = eval_python(m, test_ds)
print(f"PY Acc(after Phase-1): {py_acc:.2%}")

print("\n💾 체크포인트 저장(헤드 가중치, NumPy):", CKPT_BEFORE)
np.save(CKPT_BEFORE, m.save()['weights'].numpy())
print("✅ 저장 완료")

# -------------------------------
# SavedModel & TFLite
# -------------------------------
print("\n📦 SavedModel 저장:", SAVED_MODEL_DIR)
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
tf.saved_model.save(
    m, SAVED_MODEL_DIR,
    signatures={
        'train': m.train.get_concrete_function(),
        'infer': m.infer.get_concrete_function(),
        'save': m.save.get_concrete_function(),
        'restore': m.restore.get_concrete_function(),
        'initialize': m.initialize.get_concrete_function(),
    }
)
print("✅ SavedModel 저장 완료")

print("\n🔄 TFLite 변환 중...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)
print(f"✅ TFLite 저장 완료: {TFLITE_PATH} ({len(tflite_model)/1024/1024:.2f} MB)")

# -------------------------------
# Phase-2: TFLite 증분학습 (헤드만)
# -------------------------------
print("\n" + "="*60)
print("🚀 [Phase-2] TFLite 증분학습 (헤드만)")
print("="*60)

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
interpreter.get_signature_runner("initialize")()

train_fn   = interpreter.get_signature_runner("train")
infer_fn   = interpreter.get_signature_runner("infer")
save_fn    = interpreter.get_signature_runner("save")
restore_fn = interpreter.get_signature_runner("restore")

print("\n📂 체크포인트 복원:", CKPT_BEFORE)
restore_fn(flat_weights=np.load(CKPT_BEFORE + '.npy').astype(np.float32))
print("✅ 복원 완료")

# 리허설 준비
base_feat = MobileNetV2(include_top=False, pooling='avg', weights='imagenet'); base_feat.trainable=False
ex_mgr = ExemplarManager(base_feat, capacity_per_class=EXEMPLARS_PER_CLASS)
# 기존 데이터 일부에서 exemplar 추출
for bx, by in train_ds.take(12):
    ex_mgr.add_candidates(bx, by)
rehearsal_ds = ex_mgr.make_dataset(batch_size=BATCH_SIZE)

# 증분 데이터/리허설 데이터 이터레이터
new_iter = iter(train_ds.repeat())
reh_iter = iter(rehearsal_ds) if rehearsal_ds is not None else None

print("\n🎯 증분학습 진행...")
for i in range(INCR_NEW_STEPS):
    bx, by = next(new_iter)
    out = train_fn(x=np.array(bx, dtype=np.float32), y=np.array(by, dtype=np.float32))
    if (i+1) % 10 == 0:
        print(f"  new step {i+1:3d} | loss {float(out['loss']):.4f}")

if reh_iter is not None:
    for i in range(INCR_REHEARSAL_STEPS):
        exx, exy = next(reh_iter)
        out = train_fn(x=np.array(exx, dtype=np.float32), y=np.array(exy, dtype=np.float32))
        if (i+1) % 8 == 0:
            print(f"  repr step {i+1:3d} | loss {float(out['loss']):.4f}")

print("\n💾 업데이트 가중치 저장:", CKPT_AFTER)
np.save(CKPT_AFTER, save_fn()['weights'])
print("✅ 업데이트 완료")

# -------------------------------
# 평가 (증분 전/후)
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

print("\n" + "="*60)
print("📊 증분학습 전/후 정확도 (balanced preset)")
print("="*60)
acc_before = evaluate_with_ckpt(TFLITE_PATH, CKPT_BEFORE + '.npy', test_ds)
acc_after  = evaluate_with_ckpt(TFLITE_PATH, CKPT_AFTER  + '.npy', test_ds)
print(f"🔹 업데이트 전 정확도: {acc_before:.2%}")
print(f"🔹 업데이트 후 정확도: {acc_after:.2%}")
print("="*60)

# -------------------------------
# 7. 업데이트 전/후 비교
# -------------------------------
print("\n" + "="*60)
print("📊 업데이트 전/후 성능 비교")
print("="*60)

def evaluate_model(interpreter, checkpoint_path, test_ds):
    """특정 체크포인트 불러와서 test accuracy 평가"""
    infer_fn = interpreter.get_signature_runner("infer")
    restore_fn = interpreter.get_signature_runner("restore")
    restore_fn(checkpoint_path=np.array(checkpoint_path, dtype=np.string_))

    total, correct = 0, 0
    for test_x, test_y in test_ds:
        result = infer_fn(x=test_x.numpy())
        preds = np.argmax(result["output"], axis=1)
        labels = np.argmax(test_y.numpy(), axis=1)
        correct += np.sum(preds == labels)
        total += len(labels)
    return correct / total

# 업데이트 전/후 체크포인트 경로
ckpt_before = "D:/2025-1/friday/last/checkpoint.ckpt"
ckpt_after  = "D:/2025-1/friday/last/checkpoint_updated.ckpt"

# 새로운 인터프리터 두 개 생성 (각각 따로 restore 해야 안전)
interpreter_before = tf.lite.Interpreter(model_path="model.tflite")
interpreter_before.allocate_tensors()

interpreter_after = tf.lite.Interpreter(model_path="model.tflite")
interpreter_after.allocate_tensors()

# 평가
acc_before = evaluate_model(interpreter_before, ckpt_before, test_ds)
acc_after = evaluate_model(interpreter_after, ckpt_after, test_ds)

print(f"🔹 업데이트 전 정확도: {acc_before:.2%}")
print(f"🔹 업데이트 후 정확도: {acc_after:.2%}")
print("="*60)



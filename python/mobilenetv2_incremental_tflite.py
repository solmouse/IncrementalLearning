# -*- coding: utf-8 -*-
"""
MobileNetV2 온디바이스 증분학습 (단일 드라이브 + initialize 패치)
- 모든 파일/캐시/모델을 동일 드라이브 하위로 통일
- initialize(): 모든 변수(베이스+헤드)를 READ 없이 AssignVariableOp로 materialize
- TFLite에서 allocate_tensors() 후 반드시 initialize() 먼저 호출
- 학습 안정화: LR=3e-4, Epochs=5, 간단 증강
"""

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
# 0) 드라이브 경로 통일
# -------------------------------
def pick_root():
    for root in (r"D:\\", r"C:\\"):
        if os.path.exists(root):
            return root
    return os.getcwd()

ROOT = pick_root()
WORK_DIR  = os.path.join(ROOT, "tf_runs", "mobilenetv2_incremental")
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
# 1) 하이퍼파라미터
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
# 2) 데이터
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
    image = preprocess_input(image)  # [-1,1]
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

print("\n데이터셋 로딩 (단일 드라이브 경로 통일) ...")
zip_path = keras.utils.get_file(
    fname="cats_and_dogs_filtered.zip",
    origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
    cache_dir=DATA_DIR,
    cache_subdir="",
    extract=False
)

root = Path(zip_path).with_suffix("")
if root.exists():
    shutil.rmtree(root)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(DATA_DIR)

if not (root / "train").exists():
    # 중첩 폴더 보정
    subdirs = [p for p in Path(DATA_DIR).iterdir() if p.is_dir() and "cats_and_dogs_filtered" in p.name]
    assert len(subdirs) >= 1
    root = subdirs[0]

train_dir = str(root / "train")
val_dir   = str(root / "validation")

_raw_train = keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    label_mode="int", shuffle=True
)
_raw_val = keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    label_mode="int", shuffle=False
)

train_ds = (_raw_train
    .take(max(1, NUM_TRAIN_SAMPLES // BATCH_SIZE))
    .map(lambda x,y: preprocess(x, tf.cast(y, tf.int32), training=True), num_parallel_calls=tf.data.AUTOTUNE)
    .cache(os.path.join(CACHE_DIR, "train.cache"))
    .prefetch(tf.data.AUTOTUNE))

test_ds = (_raw_val
    .map(lambda x,y: preprocess(x, tf.cast(y, tf.int32), training=False), num_parallel_calls=tf.data.AUTOTUNE)
    .cache(os.path.join(CACHE_DIR, "val.cache"))
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
# 5) 모델 (전 변수 초기값 스냅샷 → initialize에서 Assign)
# -------------------------------
class IncrementalModel(tf.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.base = MobileNetV2(include_top=False, pooling='avg', weights='imagenet')
        self.base.trainable = False
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
            tf.keras.layers.Dense(num_classes, name='dense_2')
        ])
        # ★ 변수 생성(build)용 더미 패스
        dummy = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3], tf.float32)
        _ = self.head(self.base(dummy, training=False), training=False)

        # ★ 모든 변수 리스트 + 초기값 스냅샷(상수)
        self._var_list = list(self.base.variables) + list(self.head.variables)
        self._var_init_consts = [tf.constant(v.numpy()) for v in self._var_list]

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=LABEL_SMOOTHING)
        self.lr = tf.constant(LR, dtype=tf.float32)

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
        clipped = [tf.clip_by_norm(g, CLIP_NORM) if g is not None else None for g in grads]
        for w, g in zip(self.head.trainable_variables, clipped):
            if g is not None:
                w.assign_sub(self.lr * g)
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
        # ★ READ 없이 바로 값 주입 → READ_VARIABLE 생성 안 됨
        for v, c in zip(self._var_list, self._var_init_consts):
            tf.raw_ops.AssignVariableOp(resource=v.handle, value=c,
                                        name="init_"+v.name.replace(":","_"))
        return {"ok": True}

# -------------------------------
# 6) Python 초기 학습
# -------------------------------
m = IncrementalModel(num_classes=NUM_CLASSES)
for images, _ in train_ds.take(1):
    _ = m.infer(images)

print("\n모델 빌드 완료. 초기 학습 시작...")
for epoch in range(NUM_EPOCHS):
    losses = []
    tic = time.time()
    for step, (x, y) in enumerate(train_ds, 1):
        out = m.train(x, y)
        losses.append(float(out['loss']))
        if step % 5 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Step {step} - Loss {float(out['loss']):.4f}")
    print(f"✅ Epoch {epoch+1} 완료 | 평균 Loss: {np.mean(losses):.4f} | 경과: {time.time()-tic:.1f}s")

print("\n💾 체크포인트 저장(NumPy):", CKPT_BEFORE)
np.save(CKPT_BEFORE, m.save()['weights'].numpy())
print("✅ 저장 완료")

# -------------------------------
# 7) SavedModel 저장
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

# -------------------------------
# 8) TFLite 변환
# -------------------------------
print("\n🔄 TFLite 변환 중...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)
print(f"✅ TFLite 저장 완료: {TFLITE_PATH} ({len(tflite_model)/1024/1024:.2f} MB)")

# -------------------------------
# 9) TFLite 증분학습
# -------------------------------
print("\n" + "="*60)
print("🚀 TFLite 증분학습 테스트")
print("="*60)

# delegate가 의심되면 experimental_delegates=[]로 시도 가능
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH, experimental_delegates=[])
interpreter.allocate_tensors()
init_fn    = interpreter.get_signature_runner("initialize"); init_fn()
train_fn   = interpreter.get_signature_runner("train")
infer_fn   = interpreter.get_signature_runner("infer")
save_fn    = interpreter.get_signature_runner("save")
restore_fn = interpreter.get_signature_runner("restore")

print("\n📂 체크포인트 복원:", CKPT_BEFORE)
restore_fn(flat_weights=np.load(CKPT_BEFORE + '.npy').astype(np.float32))
print("✅ 복원 완료")

# 리허설(선택)
rehearsal_ds = None
initial_data_for_exemplars = train_ds.take(10)
incremental_update_data    = train_ds.skip(10).take(10)
if EXEMPLARS_PER_CLASS > 0:
    base = MobileNetV2(include_top=False, pooling='avg', weights='imagenet'); base.trainable=False
    ex_mgr = ExemplarManager(base, capacity_per_class=EXEMPLARS_PER_CLASS)
    for bx, by in initial_data_for_exemplars:
        ex_mgr.add_candidates(bx, by)
    rehearsal_ds = ex_mgr.make_dataset(batch_size=BATCH_SIZE)

print("\n🎯 증분학습 진행...")
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

print("\n💾 업데이트 가중치 저장:", CKPT_AFTER)
np.save(CKPT_AFTER, save_fn()['weights'])
print("✅ 업데이트 완료")

# -------------------------------
# 10) 평가 + BiC-lite
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
print("📊 증분학습 전/후 정확도 + BiC-lite")
print("="*60)
acc_before = evaluate_with_ckpt(TFLITE_PATH, CKPT_BEFORE + '.npy', test_ds)
acc_after  = evaluate_with_ckpt(TFLITE_PATH, CKPT_AFTER + '.npy',  test_ds)
print(f"🔹 업데이트 전 정확도: {acc_before:.2%}")
print(f"🔹 업데이트 후 정확도: {acc_after:.2%}")

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
print(f"🔹 BiC-lite 적용 후 정확도: {correct/max(total,1):.2%}")
print("="*60)

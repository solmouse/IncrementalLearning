# -*- coding: utf-8 -*-
"""
MobileNetV2 Class Incremental Learning (True CIL)
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

BASE_DIR = r"YOUR BASE DIR"   # TODO: 예) r"D:\tf_runs\mobilenetv2_cil"
WORK_DIR  = os.path.join(BASE_DIR, "work_true_cil")
DATA_DIR  = os.path.join(BASE_DIR, "data_true_cil")
CACHE_DIR = os.path.join(WORK_DIR, "cache")
CKPT_DIR  = os.path.join(WORK_DIR, "ckpt")
EXPORT_DIR= os.path.join(WORK_DIR, "export")

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

NUM_EPOCHS_PHASE1 = 10
WARMUP_EPOCHS_PHASE1 = 3

NUM_EPOCHS_PHASE2 = 6
STEPS_PER_EPOCH_PHASE2 = 60  # alternating new/reh steps

HEAD_LR = 1e-3
BASE_LR = 5e-5
FINE_TUNE_FRACTION = 0.25

LABEL_SMOOTHING = 0.10
CLIP_NORM = 1.0

KD_T = 2.0
KD_LAMBDA = 1.0

EXEMPLARS_PER_CLASS = 16

BIC_STEPS = 200
BIC_LR = 5e-3

PREFETCH_BUFFER = 1

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# Data utils (flowers)
# =============================================================================
FLOWER_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

def _extract_flowers():
    print("\n[Data] flowers 다운로드/압축해제...")
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
        if not imgs:
            raise RuntimeError(f"[Data] class folder empty: {src_root / cname}")
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
    image = preprocess_input(image)  # MobileNetV2: [-1,1]
    return image, label

def make_ds_from_dir(root_dir, class_names, batch_size, training,
                     subset=None, validation_split=None,
                     use_cache=False, cache_tag=None):
    """
    subset(=training/validation)으로 split을 사용할 때는
    image_dataset_from_directory 단계에서 shuffle=True 강제
    """
    use_subset = (subset is not None)
    if use_subset and validation_split is None:
        raise ValueError("subset을 쓰려면 validation_split이 필요함")

    ds = keras.preprocessing.image_dataset_from_directory(
        directory=str(root_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=True if use_subset else training,  # 중요
        seed=SEED,
        validation_split=(float(validation_split) if use_subset else None),
        subset=(subset if use_subset else None),
    )

    if use_cache:
        tag = cache_tag if cache_tag is not None else f"{Path(root_dir).name}_{subset or ('train' if training else 'val')}"
        ds = ds.cache(os.path.join(CACHE_DIR, f"{tag}.cache"))

    ds = ds.map(lambda x, y: preprocess(x, y, training=training), num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(1000, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.prefetch(PREFETCH_BUFFER)
    return ds

def make_ds_full_and_split(root_dir, class_names, batch_size, val_ratio=0.2, shuffle_buf=5000, cache_tag=None):
    """
    ✅ eval/calib용 안전 split:
    - subset/validation_split을 쓰지 않고 전체를 로드
    - (batch 단위) 셔플 후 take/skip으로 직접 분할
    => val에 특정 클래스만 몰리는 현상 방지
    """
    ds_all = keras.preprocessing.image_dataset_from_directory(
        directory=str(root_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=False,  # 여기서는 우리가 직접 shuffle할 거라 False
        seed=SEED,
    )

    # 필요하면 캐시 (cardinality 안정화에도 도움)
    if cache_tag is not None:
        ds_all = ds_all.cache(os.path.join(CACHE_DIR, f"{cache_tag}.cache"))

    ds_all = ds_all.map(lambda x, y: preprocess(x, y, training=False), num_parallel_calls=tf.data.AUTOTUNE)

    # batch-stream을 셔플(고정)한 뒤 split
    ds_all = ds_all.shuffle(shuffle_buf, seed=SEED, reshuffle_each_iteration=False)

    card = tf.data.experimental.cardinality(ds_all).numpy()
    if card < 0:
        # 드물지만 안전장치
        ds_all = ds_all.cache()
        card = tf.data.experimental.cardinality(ds_all).numpy()

    val_batches = max(1, int(card * float(val_ratio)))
    train_batches = max(1, card - val_batches)

    val_ds = ds_all.take(val_batches).prefetch(PREFETCH_BUFFER)
    train_ds = ds_all.skip(val_batches).take(train_batches).prefetch(PREFETCH_BUFFER)
    return train_ds, val_ds


# =============================================================================
# Exemplar Memory
# =============================================================================
class ExemplarMemory:
    def __init__(self, exemplars_per_class=16):
        self.m = exemplars_per_class
        self.images = None    # (N,H,W,3) float32 (preprocess_input applied)
        self.labels = None    # (N,) int32
        self.t_logits = None  # (N,C) float32
        self.class_to_indices = {}

    def __len__(self):
        return 0 if self.images is None else int(self.images.shape[0])

    def build_exemplars(self, feature_extractor, classifier_model, ds_noaug, num_classes, class_ids):
        print("\n[Memory] exemplar 구축(herding)...")

        xs, ys = [], []
        for x, y in ds_noaug:
            xs.append(x.numpy())
            ys.append(y.numpy())
        X = np.concatenate(xs, axis=0).astype(np.float32)
        Y = np.concatenate(ys, axis=0).astype(np.int32)

        feats = feature_extractor.predict(X, batch_size=BATCH_SIZE, verbose=0)
        logits = classifier_model.predict(X, batch_size=BATCH_SIZE, verbose=0).astype(np.float32)
        if logits.shape[1] != num_classes:
            raise ValueError(f"[Memory] logits dim mismatch: got {logits.shape[1]}, expected {num_classes}")

        keep_imgs, keep_lbls, keep_tlog = [], [], []
        self.class_to_indices = {}

        for cid in class_ids:
            idxs = np.where(Y == cid)[0]
            if idxs.size == 0:
                continue

            f = feats[idxs]
            mu = np.mean(f, axis=0, keepdims=True)
            dists = np.linalg.norm(f - mu, axis=1)
            order = np.argsort(dists)
            chosen_local = order[:min(self.m, len(order))]
            chosen_global = idxs[chosen_local]

            start_idx = sum(a.shape[0] for a in keep_imgs) if keep_imgs else 0
            self.class_to_indices[int(cid)] = list(range(start_idx, start_idx + len(chosen_global)))

            keep_imgs.append(X[chosen_global])
            keep_lbls.append(Y[chosen_global])
            keep_tlog.append(logits[chosen_global])

        if keep_imgs:
            self.images = np.concatenate(keep_imgs, axis=0).astype(np.float32)
            self.labels = np.concatenate(keep_lbls, axis=0).astype(np.int32)
            self.t_logits = np.concatenate(keep_tlog, axis=0).astype(np.float32)
        else:
            self.images = np.zeros((0, IMG_SIZE, IMG_SIZE, 3), np.float32)
            self.labels = np.zeros((0,), np.int32)
            self.t_logits = np.zeros((0, num_classes), np.float32)

        print(f"[Memory] 저장 exemplar: {len(self)} samples")

    def refresh_teacher_logits(self, classifier_model, num_classes):
        if self.images is None or len(self) == 0:
            return
        logits = classifier_model.predict(self.images, batch_size=BATCH_SIZE, verbose=0).astype(np.float32)
        if logits.shape[1] != num_classes:
            raise ValueError(f"[Memory] refresh logits dim mismatch: got {logits.shape[1]}, expected {num_classes}")
        self.t_logits = logits

    def make_rehearsal_ds(self, batch_size):
        if self.images is None or len(self) == 0:
            return None
        ds = tf.data.Dataset.from_tensor_slices((self.images, self.labels, self.t_logits))
        ds = ds.shuffle(min(len(self), 1000), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(PREFETCH_BUFFER)
        return ds

    def save(self, path):
        np.savez_compressed(
            path,
            m=self.m,
            images=self.images,
            labels=self.labels,
            t_logits=self.t_logits,
            class_to_indices=self.class_to_indices
        )

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        mem = ExemplarMemory(int(z["m"]))
        mem.images = z["images"]
        mem.labels = z["labels"]
        mem.t_logits = z["t_logits"]
        mem.class_to_indices = z["class_to_indices"].item()
        return mem


# =============================================================================
# Model building
# =============================================================================
def set_backbone_trainable_fraction(base: keras.Model, fraction: float):
    for l in base.layers:
        l.trainable = False

    fraction = float(fraction)
    if fraction <= 0.0:
        return

    conv_layers = [l for l in base.layers if isinstance(l, (layers.Conv2D, layers.DepthwiseConv2D))]
    if not conv_layers:
        return

    k = max(1, int(len(conv_layers) * fraction))
    for l in conv_layers[-k:]:
        l.trainable = True

def build_base(backbone_fraction: float):
    base = MobileNetV2(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg',
        weights='imagenet'
    )
    set_backbone_trainable_fraction(base, backbone_fraction)
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

def expand_head_weights(old_head: keras.Sequential, new_head: keras.Sequential, old_c: int, new_c: int):
    for i in range(len(old_head.layers) - 1):
        new_head.layers[i].set_weights(old_head.layers[i].get_weights())

    old_dense = old_head.layers[-1]
    new_dense = new_head.layers[-1]
    old_k, old_b = old_dense.get_weights()
    new_k, new_b = new_dense.get_weights()

    new_k[:, :old_c] = old_k
    new_b[:old_c] = old_b
    new_dense.set_weights([new_k, new_b])

def onehot(labels_int, num_classes):
    return tf.one_hot(tf.cast(labels_int, tf.int32), depth=num_classes, dtype=tf.float32)

def kd_kl(student_logits, teacher_logits, T):
    t = tf.nn.softmax(teacher_logits / T, axis=-1)
    s = tf.nn.log_softmax(student_logits / T, axis=-1)
    kl = tf.reduce_sum(t * (tf.math.log(tf.clip_by_value(t, 1e-8, 1.0)) - s), axis=-1)
    return kl * (T * T)


# =============================================================================
# Eval
# =============================================================================
def evaluate(model: keras.Model, ds, class_names, label_offset=0):
    """
    label_offset=0: 일반 평가
    label_offset>0: ds가 local label이면 global로 offset해서 평가
    """
    correct, total = 0, 0
    per_correct = np.zeros((len(class_names),), np.int64)
    per_total = np.zeros((len(class_names),), np.int64)

    for x, y_int in ds:
        logits = model.predict(x, verbose=0)
        pred = np.argmax(logits, axis=1).astype(np.int32)
        y = y_int.numpy().astype(np.int32) + int(label_offset)

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
    print(f"\n[Phase-1] Train on base classes only: C={num_classes} {BASE_CLASSES}")

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

    full.compile(optimizer=keras.optimizers.Adam(learning_rate=HEAD_LR),
                 loss=loss_fn, metrics=["accuracy"])

    warmup_epochs = min(WARMUP_EPOCHS_PHASE1, NUM_EPOCHS_PHASE1)
    if warmup_epochs > 0:
        print(f"\n[Phase-1] Warmup (head only) epochs={warmup_epochs}, lr={HEAD_LR}")
        full.fit(train_ds, validation_data=val_ds, epochs=warmup_epochs, verbose=1)

    ft_epochs = NUM_EPOCHS_PHASE1 - warmup_epochs
    if ft_epochs > 0:
        print(f"\n[Phase-1] Fine-tune backbone tail fraction={FINE_TUNE_FRACTION}, epochs={ft_epochs}, lr={BASE_LR}")
        set_backbone_trainable_fraction(base, FINE_TUNE_FRACTION)
        full.compile(optimizer=keras.optimizers.Adam(learning_rate=BASE_LR),
                     loss=loss_fn, metrics=["accuracy"])
        full.fit(train_ds, validation_data=val_ds,
                 epochs=warmup_epochs + ft_epochs,
                 initial_epoch=warmup_epochs,
                 verbose=1)

    acc, per_acc, per_n = evaluate(full, val_ds, BASE_CLASSES, label_offset=0)
    print(f"[Phase-1] val acc: {acc:.4f}")
    for i, c in enumerate(BASE_CLASSES):
        print(f"    {c:12s}: acc={per_acc[i]:.4f} | n={int(per_n[i])}")

    w_path = os.path.join(CKPT_DIR, "phase1.weights.h5")
    full.save_weights(w_path)

    meta_path = os.path.join(CKPT_DIR, "phase1_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"classes": BASE_CLASSES}, f, ensure_ascii=False, indent=2)

    print(f"[Phase-1] weights saved: {w_path}")
    print(f"[Phase-1] meta saved: {meta_path}")

    # memory: training split only
    mem = ExemplarMemory(exemplars_per_class=EXEMPLARS_PER_CLASS)
    mem_build_ds = make_ds_from_dir(
        subset_phase1, BASE_CLASSES, BATCH_SIZE,
        training=False, subset="training", validation_split=VAL_SPLIT,
        use_cache=False
    )
    mem.build_exemplars(feat_model, full, mem_build_ds, num_classes=num_classes,
                        class_ids=list(range(num_classes)))

    mem_path = os.path.join(CKPT_DIR, "memory_phase1.npz")
    mem.save(mem_path)
    print(f"[Phase-1] memory saved: {mem_path}")

    export_increment(full, num_classes=num_classes, old_num_classes=0, new_class_start=num_classes,
                     tag="inc0_base_only", bic_alpha=1.0, bic_beta=0.0)

    return w_path, mem_path

def load_phase1_info():
    meta_path = os.path.join(CKPT_DIR, "phase1_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["classes"]


# =============================================================================
# Phase-2 (increment)
# =============================================================================
def train_increment(src_root: Path, subset_phase2: Path, phase1_weights_h5: str, phase1_memory_npz: str):
    old_classes = load_phase1_info()
    mem = ExemplarMemory.load(phase1_memory_npz)

    old_c = len(old_classes)
    new_classes = NEW_CLASSES
    new_c = old_c + len(new_classes)

    print(f"\n[Phase-2] Increment: {old_c} -> {new_c} (add {new_classes})")

    _copy_subset(src_root, subset_phase2, new_classes)

    new_ds_train = make_ds_from_dir(
        subset_phase2, new_classes, BATCH_SIZE,
        training=True, subset="training", validation_split=VAL_SPLIT,
        use_cache=False
    )
    new_ds_val = make_ds_from_dir(
        subset_phase2, new_classes, BATCH_SIZE,
        training=False, subset="validation", validation_split=VAL_SPLIT,
        use_cache=True, cache_tag="phase2_new_val"
    )

    # Teacher (old)
    t_base, t_head, teacher_full, _ = build_keras_models(old_c, backbone_fraction=0.0)
    teacher_full.load_weights(phase1_weights_h5)
    teacher_full.trainable = False

    # Student (expanded)
    s_base = build_base(0.0)
    s_base.set_weights(t_base.get_weights())

    s_head_old = build_head(old_c)
    _ = s_head_old(tf.zeros((1, 1280), tf.float32))
    s_head_old.set_weights(t_head.get_weights())

    s_head = build_head(new_c)
    _ = s_head(tf.zeros((1, 1280), tf.float32))
    expand_head_weights(s_head_old, s_head, old_c, new_c)

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    feat = s_base(inp, training=True)
    logits = s_head(feat, training=True)
    student_full = keras.Model(inp, logits)

    rehearsal_ds = mem.make_rehearsal_ds(batch_size=BATCH_SIZE)
    if rehearsal_ds is None:
        raise RuntimeError("rehearsal memory empty")

    loss_ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=LABEL_SMOOTHING)

    opt_head = tf.keras.optimizers.Adam(learning_rate=HEAD_LR)
    # ✅ backbone 처음엔 frozen이라 base optimizer를 일단 만들어도 되고, unfreeze 때 새로 만들어도 됨
    opt_base = tf.keras.optimizers.Adam(learning_rate=BASE_LR)

    head_vars = s_head.trainable_variables
    base_vars = []  # 처음엔 frozen

    def pad_logits_np(tlog_np: np.ndarray):
        if tlog_np.shape[1] == new_c:
            return tlog_np
        pad = np.zeros((tlog_np.shape[0], new_c - tlog_np.shape[1]), np.float32)
        return np.concatenate([tlog_np, pad], axis=1)

    # ✅ eager train step (unfreeze/optimizer 교체 안정)
    def train_step(x, y_onehot, t_logits_padded):
        nonlocal base_vars, opt_base

        with tf.GradientTape() as tape:
            s_logits = student_full(x, training=True)
            ce = loss_ce(y_onehot, s_logits)

            kd = tf.constant(0.0, tf.float32)
            if old_c > 0:
                kd_vec = kd_kl(s_logits[:, :old_c], t_logits_padded[:, :old_c], KD_T)
                kd = tf.reduce_mean(kd_vec)

            loss = ce + KD_LAMBDA * kd

        vars_ = head_vars + base_vars
        grads = tape.gradient(loss, vars_)
        grads = [tf.clip_by_norm(g, CLIP_NORM) if g is not None else None for g in grads]

        g_head = grads[:len(head_vars)]
        g_base = grads[len(head_vars):]

        opt_head.apply_gradients([(g, v) for g, v in zip(g_head, head_vars) if g is not None])
        if base_vars:
            opt_base.apply_gradients([(g, v) for g, v in zip(g_base, base_vars) if g is not None])

        return loss, ce, kd

    new_iter = iter(new_ds_train.repeat())
    reh_iter = iter(rehearsal_ds.repeat())

    total_steps = STEPS_PER_EPOCH_PHASE2 * NUM_EPOCHS_PHASE2
    unfreeze_step = int(total_steps * 0.33)

    print(f"[Phase-2] training steps: {total_steps} (steps/epoch={STEPS_PER_EPOCH_PHASE2})")

    t0 = time.time()
    for step in range(1, total_steps + 1):
        if step == unfreeze_step:
            print(f"\n[Phase-2] Unfreeze backbone tail fraction={FINE_TUNE_FRACTION} at step={step}")
            set_backbone_trainable_fraction(s_base, FINE_TUNE_FRACTION)

            # ✅ trainable var 세트가 바뀌므로 base optimizer는 새로 만들어서 안전하게
            opt_base = tf.keras.optimizers.Adam(learning_rate=BASE_LR)
            base_vars = s_base.trainable_variables  # 이제 비어있지 않음

        if step % 2 == 1:
            x, y_local = next(new_iter)
            y_global = y_local + old_c
            y = onehot(y_global, new_c)

            tlog = teacher_full.predict(x, verbose=0).astype(np.float32)
            tlog = pad_logits_np(tlog)
        else:
            x, y_global, tlog_stored = next(reh_iter)
            y = onehot(y_global, new_c)
            tlog = pad_logits_np(tlog_stored.numpy().astype(np.float32))

        loss, ce, kd = train_step(x, y, tf.convert_to_tensor(tlog, tf.float32))

        if step % 50 == 0:
            print(f"  step {step:4d}/{total_steps} | loss {float(loss):.4f} (ce {float(ce):.4f}, kd {float(kd):.4f})")

    print(f"[Phase-2] done. time={time.time()-t0:.1f}s")

    # evaluate all classes (validation) - ✅ 안전 split 적용
    subset_all = Path(DATA_DIR) / "flow_subset_all"
    _copy_subset(src_root, subset_all, CLASS_ORDER)

    calib_ds, eval_ds = make_ds_full_and_split(
        subset_all, CLASS_ORDER, BATCH_SIZE,
        val_ratio=VAL_SPLIT,
        shuffle_buf=8000,
        cache_tag="all_eval_full"
    )

    acc_all, per_acc_all, per_n_all = evaluate(student_full, eval_ds, CLASS_ORDER, label_offset=0)
    print(f"[Phase-2] eval(all/val) acc: {acc_all:.4f}")
    for i, c in enumerate(CLASS_ORDER):
        print(f"    {c:12s}: acc={per_acc_all[i]:.4f} | n={int(per_n_all[i])}")

    # update memory logits
    mem.refresh_teacher_logits(student_full, num_classes=new_c)
    mem_path2 = os.path.join(CKPT_DIR, "memory_after_inc1.npz")
    mem.save(mem_path2)
    print(f"[Phase-2] memory updated: {mem_path2}")

    # save weights/meta
    w_path2 = os.path.join(CKPT_DIR, "inc1.weights.h5")
    student_full.save_weights(w_path2)

    meta_path2 = os.path.join(CKPT_DIR, "inc1_meta.json")
    with open(meta_path2, "w", encoding="utf-8") as f:
        json.dump({"classes": CLASS_ORDER[:new_c]}, f, ensure_ascii=False, indent=2)

    print(f"[Phase-2] weights saved: {w_path2}")
    print(f"[Phase-2] meta saved: {meta_path2}")

    # BiC calibration (✅ 같은 split의 train쪽 = calib_ds 사용)
    alpha, beta = bic_calibrate(student_full, calib_ds, old_c, new_c)

    export_increment(student_full, num_classes=new_c, old_num_classes=old_c, new_class_start=old_c,
                     tag="inc1_after_add", bic_alpha=float(alpha), bic_beta=float(beta))

    # debug: new-class-only val acc (global label offset 반영)
    new_val_acc, new_per_acc, new_per_n = evaluate(student_full, new_ds_val, new_classes, label_offset=old_c)
    print(f"[Phase-2] (debug) new-class-only val acc (global): {new_val_acc:.4f}")
    for i, c in enumerate(new_classes):
        print(f"    {c:12s}: acc={new_per_acc[i]:.4f} | n={int(new_per_n[i])}")

    return w_path2, mem_path2


def bic_calibrate(student_full: keras.Model, calib_ds, old_c, new_c):
    print("\n[BiC] calibration...")
    alpha = tf.Variable(1.0, trainable=True, dtype=tf.float32)
    beta  = tf.Variable(0.0, trainable=True, dtype=tf.float32)
    opt = tf.keras.optimizers.SGD(learning_rate=BIC_LR)

    @tf.function
    def step(x, y_int):
        y = onehot(y_int, new_c)
        with tf.GradientTape() as tape:
            logits = student_full(x, training=False)
            old = logits[:, :old_c]
            new = logits[:, old_c:]
            logits_corr = tf.concat([old, alpha * new + beta], axis=1)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits_corr, from_logits=True))
        grads = tape.gradient(loss, [alpha, beta])
        opt.apply_gradients(zip(grads, [alpha, beta]))
        return loss

    it = iter(calib_ds.repeat())
    for s in range(1, BIC_STEPS + 1):
        x, y = next(it)
        loss = step(x, y)
        if s % 50 == 0:
            print(f"  bic step {s}/{BIC_STEPS} | loss {float(loss):.4f} | alpha {float(alpha):.4f} beta {float(beta):.4f}")

    print(f"[BiC] done: alpha={float(alpha):.4f}, beta={float(beta):.4f}")
    return alpha.numpy(), beta.numpy()


# =============================================================================
# Export module (SavedModel 3 signatures + TFLite infer-only)
# =============================================================================
def make_trainable_module(num_classes, old_num_classes, new_class_start):
    class CILModule(tf.Module):
        def __init__(self):
            super().__init__()
            self.num_classes = int(num_classes)
            self.old_num_classes = int(old_num_classes)
            self.new_class_start = int(new_class_start)

            self.base = build_base(FINE_TUNE_FRACTION)
            self.head = build_head(self.num_classes)

            self.bic_alpha = tf.Variable(1.0, trainable=True, dtype=tf.float32, name="bic_alpha")
            self.bic_beta  = tf.Variable(0.0, trainable=True, dtype=tf.float32, name="bic_beta")

            self.loss_ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=LABEL_SMOOTHING)
            self.opt_head = tf.keras.optimizers.Adam(learning_rate=HEAD_LR)
            self.opt_base = tf.keras.optimizers.Adam(learning_rate=BASE_LR)
            self.opt_bic  = tf.keras.optimizers.SGD(learning_rate=BIC_LR)

        def _apply_bic(self, logits):
            if self.new_class_start >= self.num_classes:
                return logits
            old = logits[:, :self.new_class_start]
            new = logits[:, self.new_class_start:]
            new_corr = self.bic_alpha * new + self.bic_beta
            return tf.concat([old, new_corr], axis=1)

        @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)])
        def infer(self, x):
            feat = self.base(x, training=False)
            logits = self.head(feat, training=False)
            logits = self._apply_bic(logits)
            return {"logits": logits, "probs": tf.nn.softmax(logits)}

        @tf.function(input_signature=[
            tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32),
            tf.TensorSpec([None, num_classes], tf.float32),
            tf.TensorSpec([None, num_classes], tf.float32),
            tf.TensorSpec([None, 1], tf.float32),
        ])
        def train(self, x, y, t_logits, t_mask):
            head_vars = self.head.trainable_variables
            base_vars = [v for v in self.base.trainable_variables if v.trainable]

            with tf.GradientTape() as tape:
                feat = self.base(x, training=False)
                s_logits = self.head(feat, training=False)

                ce = self.loss_ce(y, s_logits)

                kd = 0.0
                if self.old_num_classes > 0:
                    s_old = s_logits[:, :self.old_num_classes]
                    t_old = t_logits[:, :self.old_num_classes]
                    kd_vec = kd_kl(s_old, t_old, KD_T)
                    kd = tf.reduce_mean(kd_vec * tf.squeeze(t_mask, axis=1))

                loss = ce + KD_LAMBDA * kd

            vars_ = head_vars + base_vars
            grads = tape.gradient(loss, vars_)
            grads = [tf.clip_by_norm(g, CLIP_NORM) if g is not None else None for g in grads]

            g_head = grads[:len(head_vars)]
            g_base = grads[len(head_vars):]

            self.opt_head.apply_gradients([(g, v) for g, v in zip(g_head, head_vars) if g is not None])
            if base_vars:
                self.opt_base.apply_gradients([(g, v) for g, v in zip(g_base, base_vars) if g is not None])

            return {"loss": loss, "ce": ce, "kd": kd}

        @tf.function(input_signature=[
            tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32),
            tf.TensorSpec([None, num_classes], tf.float32),
        ])
        def train_bic(self, x, y):
            if self.new_class_start >= self.num_classes:
                return {"loss": tf.constant(0.0, tf.float32), "alpha": self.bic_alpha, "beta": self.bic_beta}

            with tf.GradientTape() as tape:
                feat = self.base(x, training=False)
                logits = self.head(feat, training=False)
                logits = self._apply_bic(logits)
                loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True))

            grads = tape.gradient(loss, [self.bic_alpha, self.bic_beta])
            grads = [
                tf.zeros_like(self.bic_alpha) if grads[0] is None else grads[0],
                tf.zeros_like(self.bic_beta)  if grads[1] is None else grads[1],
            ]
            self.opt_bic.apply_gradients(zip(grads, [self.bic_alpha, self.bic_beta]))
            return {"loss": loss, "alpha": self.bic_alpha, "beta": self.bic_beta}

    return CILModule()


def export_increment(student_full: keras.Model,
                     num_classes, old_num_classes, new_class_start,
                     tag, bic_alpha, bic_beta):
    print(f"\n[Export] {tag} ...")
    export_path = os.path.join(EXPORT_DIR, tag)
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    os.makedirs(export_path, exist_ok=True)

    module = make_trainable_module(
        num_classes=num_classes,
        old_num_classes=old_num_classes,
        new_class_start=new_class_start
    )

    dummy = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), tf.float32)
    _ = module.infer(dummy)

    base_layer = None
    head_layer = None
    for l in student_full.layers:
        if isinstance(l, keras.Model) and "mobilenet" in l.name.lower():
            base_layer = l
        if isinstance(l, keras.Sequential) and l.name == "head":
            head_layer = l

    if base_layer is None or head_layer is None:
        for l in student_full.layers:
            if isinstance(l, keras.Model) and base_layer is None:
                base_layer = l
            if isinstance(l, keras.Sequential) and head_layer is None:
                head_layer = l
        if base_layer is None or head_layer is None:
            base_layer = student_full.layers[1]
            head_layer = student_full.layers[2]

    module.base.set_weights(base_layer.get_weights())
    module.head.set_weights(head_layer.get_weights())
    module.bic_alpha.assign(float(bic_alpha))
    module.bic_beta.assign(float(bic_beta))

    # optimizer slot build (안전)
    try:
        module.opt_head.build(module.head.trainable_variables)
    except Exception:
        pass
    try:
        base_vars = [v for v in module.base.trainable_variables if v.trainable]
        if base_vars:
            module.opt_base.build(base_vars)
    except Exception:
        pass
    try:
        module.opt_bic.build([module.bic_alpha, module.bic_beta])
    except Exception:
        pass

    # SavedModel: 3시그니처 유지
    tf.saved_model.save(
        module, export_path,
        signatures={"infer": module.infer, "train": module.train, "train_bic": module.train_bic}
    )
    print(f"[Export] SavedModel saved: {export_path}")

    # TFLite: infer-only
    converter = tf.lite.TFLiteConverter.from_saved_model(
        export_path,
        signature_keys=["infer"]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()

    tflite_path = os.path.join(export_path, f"{tag}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite)
    print(f"[Export] TFLite (infer-only) saved: {tflite_path}")

    meta = {
        "tag": tag,
        "num_classes": int(num_classes),
        "old_num_classes": int(old_num_classes),
        "new_class_start": int(new_class_start),
        "bic_alpha": float(bic_alpha),
        "bic_beta": float(bic_beta),
        "classes": CLASS_ORDER[:int(num_classes)],
        "tflite_signatures": ["infer"],
        "savedmodel_signatures": ["infer", "train", "train_bic"],
    }
    with open(os.path.join(export_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================
def main():
    src_root = _extract_flowers()

    subset_phase1 = Path(DATA_DIR) / "flow_subset_phase1_base"
    subset_phase2 = Path(DATA_DIR) / "flow_subset_phase2_new"

    w0, mem0 = train_phase1(src_root, subset_phase1)
    train_increment(src_root, subset_phase2, w0, mem0)

    print("\nDone.")


if __name__ == "__main__":
    main()

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import time

IMG_SIZE = 224
INIT_CLASSES = 2
SAVED_MODEL_DIR = ""
BATCH_SIZE = 32
data_dir = ""

# CPU용
NUM_TRAIN_SAMPLES = 500
NUM_EPOCHS = 2

# -------------------------------
# MobileNetV2 버리고 단순 CNN 사용 (공식 예제 방식)
# -------------------------------
class Model(tf.Module):
    def __init__(self, num_classes=INIT_CLASSES):
        super().__init__()
        
        # 💡 단순하지만 강력한 CNN
        self.model = tf.keras.Sequential([
            # 입력: (224, 224, 3)
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            tf.keras.layers.MaxPooling2D(2),
            # (111, 111, 32)
            
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            # (54, 54, 64)
            
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            # (26, 26, 128)
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
            tf.keras.layers.Dense(num_classes, name='dense_2')
        ])
        
        self.model.compile(
            optimizer='sgd',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        )
    
    @tf.function(input_signature=[
    tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32),
    tf.TensorSpec([None, None], tf.float32)
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        lr = 0.001
        # optimizer.apply_gradients 대신 직접 업데이트
        for w, g in zip(self.model.trainable_variables, grads):
            w.assign_sub(lr * g)

        return {"loss": loss}


    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)
    ])
    def infer(self, x):
        logits = self.model(x, training=False)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return {
            "output": probabilities,
            "logits": logits
        }
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = []
        tensors_to_save = []
        for layer in self.model.layers:
            for w in layer.weights:
                # 고유 이름 부여
                unique_name = layer.name + "/" + w.name
                tensor_names.append(unique_name)
                tensors_to_save.append(w)

        tf.raw_ops.Save(
            filename=checkpoint_path,
            tensor_names=tensor_names,
            data=tensors_to_save,
            name='save'
        )
        return {"checkpoint_path": checkpoint_path}

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for layer in self.model.layers:
            for w in layer.weights:
                unique_name = layer.name + "/" + w.name
                restored = tf.raw_ops.Restore(
                    file_pattern=checkpoint_path,
                    tensor_name=unique_name,
                    dt=w.dtype,
                    name='restore'
                )
                restored.set_shape(w.shape)
                w.assign(restored)
                restored_tensors[unique_name] = restored
        return restored_tensors

# -------------------------------
# 1. 모델 생성 및 변수 확인
# -------------------------------
m = Model(num_classes=INIT_CLASSES)

print("="*60)
print("모델 구조:")
m.model.summary()
print(f"\n총 파라미터: {m.model.count_params():,}")
print(f"학습 가능한 변수 수: {len(m.model.trainable_variables)}")
print("="*60)

# -------------------------------
# 2. 데이터셋 로드 및 전처리
# -------------------------------
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, INIT_CLASSES)
    return image, label

print("\n데이터셋 로딩 중...")
train_dataset, test_dataset = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    data_dir=data_dir
)

train_ds = (train_dataset
    .take(NUM_TRAIN_SAMPLES)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .shuffle(500)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (test_dataset
    .take(100)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

for images, labels in train_ds.take(1):
    print(f"이미지 배치 shape: {images.shape}")
    print(f"라벨 배치 shape: {labels.shape}")

# -------------------------------
# 3. 초기 학습
# -------------------------------
print("\n초기 학습 시작...")

for epoch in range(NUM_EPOCHS):
    losses = []
    batch_count = 0
    start_time = time.time()
    
    for batch in train_ds:
        x, y = batch
        result = m.train(x, y)
        losses.append(result['loss'].numpy())
        batch_count += 1
        
        if batch_count % 5 == 0:
            print(f"  Batch {batch_count}/{NUM_TRAIN_SAMPLES//BATCH_SIZE}, Loss: {result['loss'].numpy():.4f}")
    
    elapsed = time.time() - start_time
    print(f"✅ Epoch {epoch+1}/{NUM_EPOCHS} 완료 - 평균 Loss: {np.mean(losses):.4f}, 소요시간: {elapsed:.1f}초")

# 체크포인트 저장
checkpoint_path = 'D:/2025-1/friday/last/checkpoint.ckpt'
m.save(checkpoint_path=np.array(checkpoint_path, dtype=np.string_))
print(f"\n💾 체크포인트 저장: {checkpoint_path}")

# -------------------------------
# 4. SavedModel 저장
# -------------------------------
print(f"\n📦 SavedModel 저장 중...")
tf.saved_model.save(
    m,
    SAVED_MODEL_DIR,
    signatures={
        'train': m.train.get_concrete_function(),
        'infer': m.infer.get_concrete_function(),
        'save': m.save.get_concrete_function(),
        'restore': m.restore.get_concrete_function(),
    }
)
print(f"✅ SavedModel 저장 완료: {SAVED_MODEL_DIR}")

# -------------------------------
# 5. TFLite 변환
# -------------------------------
print("\n🔄 TFLite 변환 중...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.experimental_enable_resource_variables = True

tflite_model = converter.convert()

tflite_path = "model.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"✅ TFLite 저장 완료: {tflite_path} ({len(tflite_model)/1024/1024:.2f} MB)")

# -------------------------------
# 6. TFLite 증분학습 테스트
# -------------------------------
print("\n" + "="*60)
print("🚀 TFLite 증분 학습 테스트")
print("="*60)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# 모든 텐서 확인
for detail in interpreter.get_tensor_details():
    print(f"이름: {detail['name']}")
    print(f"  dtype: {detail['dtype']}, shape: {detail['shape']}")
    print(f"  index: {detail['index']}")
    print()

print("\n사용 가능한 시그니처:")
for sig in interpreter.get_signature_list():
    print(f"  ✓ {sig}")

train_fn = interpreter.get_signature_runner("train")
infer_fn = interpreter.get_signature_runner("infer")
save_fn = interpreter.get_signature_runner("save")
restore_fn = interpreter.get_signature_runner("restore")

# 체크포인트 복원
print("\n📂 체크포인트 복원 중...")
restore_fn(checkpoint_path=np.array(checkpoint_path, dtype=np.string_))
print("✅ 체크포인트 복원 완료!")

# 증분 학습
print("\n🎯 증분 학습 시작...")
step_count = 0
for batch in train_ds.take(10):
    new_x, new_y = batch
    try:
        result = train_fn(x=new_x.numpy(), y=new_y.numpy())
        loss_val = float(result['loss'])   # ✅ 안전하게 변환
        step_count += 1
        print(f"  ✓ Step {step_count}, Loss: {loss_val:.4f}")
    except Exception as e:
        print(f"  ✗ Step {step_count+1} 실패: {e}")
        import traceback
        traceback.print_exc()
        break

if step_count > 0:
    # 업데이트된 체크포인트 저장
    new_checkpoint = 'D:/2025-1/friday/last/checkpoint_updated.ckpt'
    print(f"\n💾 업데이트된 가중치 저장: {new_checkpoint}")
    save_fn(checkpoint_path=np.array(new_checkpoint, dtype=np.string_))


# 추론 테스트
print("\n🔮 추론 테스트...")
for test_batch in test_ds.take(1):
    test_x, test_y = test_batch
    result = infer_fn(x=test_x.numpy())
    predictions = np.argmax(result['output'], axis=1)
    true_labels = np.argmax(test_y.numpy(), axis=1)
    
    accuracy = np.mean(predictions == true_labels)
    print(f"  예측 shape: {result['output'].shape}")
    print(f"  예측 클래스: {predictions[:10]}")
    print(f"  실제 클래스: {true_labels[:10]}")
    print(f"  ✅ 정확도: {accuracy:.2%}")


print("\n" + "="*60)
print("🎉 완료! TFLite 온디바이스 학습 성공!")
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



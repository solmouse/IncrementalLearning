import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="../model/model.tflite")
interpreter.allocate_tensors()

train_fn = interpreter.get_signature_runner("train")
infer_fn = interpreter.get_signature_runner("infer")
save_fn = interpreter.get_signature_runner("save")
restore_fn = interpreter.get_signature_runner("restore")

restore_fn(checkpoint_path=np.array("../model/checkpoint.ckpt", dtype=np.string_))
print("checkpoint loaded")

x = np.random.rand(2, 224, 224, 3).astype(np.float32)
y = np.array([[1, 0], [0, 1]], dtype=np.float32)

for step in range(3):
    res = train_fn(x=x, y=y)
    print(f"Step {step+1}, loss={float(res['loss']):.4f}")

save_fn(checkpoint_path=np.array("../model/saved_ckpt_by_test/checkpoint_updated_new.ckpt", dtype=np.string_))
print("checkpoint updated")

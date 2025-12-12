import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="../model/model.tflite")
interpreter.allocate_tensors()

train_fn = interpreter.get_signature_runner("train")
infer_fn = interpreter.get_signature_runner("infer")
save_fn = interpreter.get_signature_runner("save")
restore_fn = interpreter.get_signature_runner("restore")

restore_fn(checkpoint_path=np.array("../model/checkpoint.ckpt", dtype=np.string_))
print("checkpoint is called")

dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
result = infer_fn(x=dummy_input)
print("expectation result:", result["output"])

import tensorflow as tf
import numpy as np

h5path = r"C:\Users\buiti\Downloads\face_classification-master\face_classification-master\trained_models\emotion_models\fer2013_mini_XCEPTION.110-0.65.hdf5"
tfpath = "model_int8.tflite"
model = tf.keras.models.load_model(h5path, compile=False)
img_size = 64, 64
#model.summary()
datasetpath = 
def preprocess_input(path)
def representative_data_gen():
    for _ in range(100):
        data = np.random.randint((1, 64, 64, 1), dtype=np.uint8)
        print(data.shape)
        yield [data.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()
open(tfpath, "wb").write(tflite_model)
print("Saved:", tfpath, "size:", len(tflite_model)/1024, "KB")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
# Get the current directory
current_dir = os.getcwd()

# Loop through all_without_eac files in the current directory
for filename in os.listdir(current_dir):
    if filename.endswith('.keras'):
        # Load the model
        model = tf.keras.models.load_model(filename)

        # Convert the model to TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the converted model
        tflite_filename = filename.replace('.keras', '.tflite')
        with open(tflite_filename, 'wb') as f:
            f.write(tflite_model)

        print(f"Converted {filename} to {tflite_filename}")

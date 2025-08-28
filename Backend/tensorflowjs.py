import tensorflow as tf
import tensorflowjs as tfjs

# Path to your trained Keras model (.h5 file)
keras_model_path = "./MLP__treatment_prediction_dataset__20250814-165920_model.h5"

# Output folder for TensorFlow.js model
tfjs_target_dir = "./tfjs_model"

# Load the Keras model
model = tf.keras.models.load_model(keras_model_path)

# Convert and save as TensorFlow.js Layers model
tfjs.converters.save_keras_model(model, tfjs_target_dir)

print(f"Model successfully converted and saved to: {tfjs_target_dir}")

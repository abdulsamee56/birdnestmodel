import tensorflow as tf
import numpy as np
from PIL import Image

# Path to the .tflite model
tflite_model_path = "./models/twobirds.tflite"

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print details to understand model input/output
print("Input Details:", input_details)
print("Output Details:", output_details)

# Helper function to preprocess image
def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert("RGB")  # Convert to RGB
    image = image.resize((input_shape[1], input_shape[2]))  # Resize to input shape
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Test the model with a sample image
image_path = "./images/bobolink.jpg"  # Replace with your test image path
input_shape = input_details[0]['shape']  # Get expected input shape
input_data = preprocess_image(image_path, input_shape)

# Set the tensor input
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the prediction results
output_data = interpreter.get_tensor(output_details[0]['index'])

# Interpret the results
print("Raw Output:", output_data)

# Example: If the model returns probabilities or logits, find the class
predicted_class = np.argmax(output_data, axis=-1)
print(f"Predicted Class: {predicted_class}")

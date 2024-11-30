import logging
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to the trained model and test image
MODEL_PATH = "./models/best.pt"  # Update this if the path to best.pt is different
IMAGE_PATH = "./images/bobolink5.jpg"  # Path to the image

def predict_species(model_path, image_path):
    """
    Load the trained model and predict the species of the bird in the image.

    Args:
        model_path (str): Path to the trained YOLO model.
        image_path (str): Path to the bird image.

    Returns:
        None
    """
    try:
        # Load the trained YOLO model
        logging.info("Loading the trained model...")
        model = YOLO(model_path)

        # Run prediction on the input image
        logging.info(f"Running prediction on {image_path}...")
        results = model.predict(image_path, save=False, verbose=False)

        # Get the top prediction (class label and confidence score)
        prediction = results[0]
        predicted_class = prediction.probs.top1  # Get the index of the top-1 predicted class
        confidence = prediction.probs.top1conf  # Get the confidence of the top-1 prediction

        # Map the class index to the species name
        species_name = model.names[predicted_class]  # YOLO stores class names in model.names

        # Log the results
        logging.info(f"Predicted Species: {species_name} (Confidence: {confidence:.2f})")

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")


if __name__ == "__main__":
    # Run the prediction function
    predict_species(MODEL_PATH, IMAGE_PATH)

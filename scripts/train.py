import os
from ultralytics import YOLO

# Define paths and parameters
MODEL_PATH = "yolo11x-cls.pt"  # Path to the YOLO model
DATA_DIR = "C:/Users/samee/Documents/cubtrain/datasets/cub2002011/images_split"  # Path to your dataset
EPOCHS = 30
IMAGE_SIZE = 640
SAVE_DIR = "C:/Users/samee/Documents/cubtrain/models"  # Path to save the model

if __name__ == "__main__":
    # Ensure the models directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load the YOLO model
    model = YOLO(MODEL_PATH)

    # Train the model
    results = model.train(data=DATA_DIR, epochs=EPOCHS, imgsz=IMAGE_SIZE)

    # Save the best model
    best_model_path = os.path.join(SAVE_DIR, "best.pt")
    model.save(best_model_path)

    print(f"Training complete. Model saved to {best_model_path}")

import torch

# Path to the model file
MODEL_PATH = "../models/best.pt"

def check_model_keys(model_path):
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        print("Keys in the model checkpoint:", checkpoint.keys())
    except Exception as e:
        print(f"Error loading model: {e}")

check_model_keys(MODEL_PATH)

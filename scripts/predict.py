from ultralytics import YOLO
from PIL import Image, ImageDraw

def predict_image(model_path, input_image, output_image):
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    print(f"Running prediction on {input_image}...")
    results = model(input_image)
    result = results[0]

    # Open the image and draw predictions
    image = Image.open(input_image)
    draw = ImageDraw.Draw(image)

    class_label = result.probs.top1
    confidence = result.probs.top1conf.item()
    text = f"{model.names[class_label]} ({confidence:.2f})"
    draw.text((10, 10), text, fill="black")

    image.save(output_image)
    print(f"Prediction saved to {output_image}")

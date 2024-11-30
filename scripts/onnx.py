import torch
import onnx
from model import 'models\best.pt'  # Replace with your model's import path

# Load the PyTorch model
model = YourModel()
model.load_state_dict(torch.load("models\best.pt"))
model.eval()

# Define dummy input matching the model's input size
dummy_input = torch.randn(1, 3, 224, 224)  # Example: batch_size=1, 3 color channels, 224x224 image

# Export to ONNX
onnx_file_path = "model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    export_params=True,
    opset_version=11,  # Use a compatible ONNX opset
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print(f"ONNX model saved to {onnx_file_path}")

import torch

print("PyTorch version:", torch.__version__)
print("Is CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device being used:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

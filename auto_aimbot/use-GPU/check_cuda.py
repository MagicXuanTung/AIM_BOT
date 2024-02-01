import torch

# Check PyTorch version
print("PyTorch Version:", torch.__version__)

# Check CUDA availability
if torch.cuda.is_available():
    print("CUDA is available.")
    # Get CUDA version
    print("CUDA Version:", torch.version.cuda)
    # Get cuDNN version
    print("cuDNN Version:", torch.backends.cudnn.version())
    # Get the current CUDA device
    print("Current CUDA Device:", torch.cuda.current_device())
    # Get the name of the current CUDA device
    print("Current CUDA Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available.")

import torch
import sys

def check_cuda():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Create test tensors
        x = torch.randn(2, 3).cuda()
        y = torch.randn(2, 3).cuda()
        z = x + y
        
        print("\nCUDA Test Computation:")
        print(f"x = {x}")
        print(f"y = {y}")
        print(f"z = x + y = {z}")
        
        print("\nCUDA Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("CUDA is not available on this system.")

if __name__ == "__main__":
    check_cuda()

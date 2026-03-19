import torch

print("=" * 50)
print("PyTorch 版本信息")
print("=" * 50)

# PyTorch 版本
print(f"PyTorch 版本：{torch.__version__}")

# CUDA 是否可用
print(f"CUDA 是否可用：{torch.cuda.is_available()}")

# CUDA 版本 (PyTorch 编译时的 CUDA 版本)
print(f"CUDA 版本：{torch.version.cuda}")

# cuDNN 版本
print(f"cuDNN 版本：{torch.backends.cudnn.version()}")

# GPU 信息
if torch.cuda.is_available():
    print(f"GPU 数量：{torch.cuda.device_count()}")
    print(f"当前 GPU：{torch.cuda.current_device()}")
    print(f"GPU 名称：{torch.cuda.get_device_name(0)}")
    print(f"GPU 显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
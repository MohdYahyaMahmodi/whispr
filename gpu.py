import torch
import sys
import platform
import os

def check_cuda():
    print("=" * 50)
    print("CUDA DETECTION TEST")
    print("=" * 50)
    
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        # List all available GPUs
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Compute capability: {props.major}.{props.minor}")
            print(f"  - Total memory: {props.total_memory / 1024 / 1024 / 1024:.2f} GB")
        
        # Test a simple CUDA operation
        print("\nTesting CUDA with a simple tensor operation...")
        try:
            x = torch.rand(5, 3).cuda()
            y = torch.rand(5, 3).cuda()
            z = x + y
            print("✓ CUDA tensor operation successful!")
        except Exception as e:
            print(f"✗ CUDA tensor operation failed: {e}")
    else:
        print("\nWhy CUDA might not be available:")
        print("1. PyTorch was installed without CUDA support")
        print("2. NVIDIA drivers are not installed or are outdated")
        print("3. CUDA is not installed or is incompatible")
        print("4. You don't have an NVIDIA GPU")
        
        # Check if NVIDIA GPU is detected by the system
        print("\nChecking for NVIDIA GPUs in the system...")
        if platform.system() == "Windows":
            # On Windows
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("✓ NVIDIA GPU detected by system, but PyTorch can't use it.")
                    print("\nOutput from nvidia-smi:")
                    print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
                else:
                    print("✗ NVIDIA GPU not detected by system (nvidia-smi failed).")
            except:
                print("✗ Could not run nvidia-smi. NVIDIA drivers might not be installed.")
        elif platform.system() == "Linux":
            # On Linux
            try:
                lspci_output = os.popen('lspci | grep -i nvidia').read()
                if lspci_output:
                    print("✓ NVIDIA GPU detected by system, but PyTorch can't use it.")
                    print("\nOutput from lspci:")
                    print(lspci_output[:500] + "..." if len(lspci_output) > 500 else lspci_output)
                else:
                    print("✗ NVIDIA GPU not detected by system.")
            except:
                print("✗ Could not check for NVIDIA GPU.")
    
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    
    if not cuda_available:
        print("To enable GPU support:")
        print("\n1. Install the appropriate NVIDIA drivers for your GPU")
        print("   Visit: https://www.nvidia.com/Download/index.aspx")
        
        print("\n2. Reinstall PyTorch with CUDA support:")
        print("   pip uninstall -y torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117")
        
        print("\n3. If using Conda:")
        print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia")
        
        print("\n4. Verify CUDA installation:")
        print("   Windows: nvidia-smi")
        print("   Linux: nvidia-smi or lspci | grep -i nvidia")
    else:
        print("Your CUDA setup is working correctly with PyTorch!")
        print("You should be able to use GPU acceleration for Whisper transcription.")

if __name__ == "__main__":
    check_cuda()
#!/usr/bin/env python3
"""
GPU Test Script for Host System
Verifies CUDA availability and GPU performance
"""

import torch
import numpy as np
import time

def test_gpu_availability():
    """Test basic GPU availability"""
    print("🔍 GPU Availability Test")
    print("=" * 50)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'✅ Yes' if cuda_available else '❌ No'}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"GPU Count: {device_count}")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f}GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    
    return cuda_available

def test_tensor_operations():
    """Test basic tensor operations on GPU"""
    print("\n🧮 Tensor Operations Test")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping tensor test")
        return False
    
    device = torch.device('cuda:0')
    
    try:
        # Create test tensors
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        # Time matrix multiplication
        start_time = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # Wait for GPU computation
        end_time = time.time()
        
        print(f"✅ Matrix multiplication (1000x1000): {(end_time - start_time)*1000:.2f}ms")
        
        # Test memory allocation
        memory_allocated = torch.cuda.memory_allocated() / 1e6
        print(f"✅ GPU Memory Allocated: {memory_allocated:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False

def test_neural_network():
    """Test neural network operations"""
    print("\n🧠 Neural Network Test")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping neural network test")
        return False
    
    device = torch.device('cuda:0')
    
    try:
        # Simple CNN for testing
        import torch.nn as nn
        
        class TestCNN(nn.Module):
            def __init__(self):
                super(TestCNN, self).__init__()
                self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((8, 8))
                self.fc = nn.Linear(64 * 8 * 8, 6)
            
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        # Create and test model
        model = TestCNN().to(device)
        model.eval()
        
        # Test input (simulating RGBD image)
        test_input = torch.randn(1, 4, 480, 640, device=device)
        
        # Time forward pass
        start_time = time.time()
        with torch.no_grad():
            output = model(test_input)
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"✅ CNN Forward Pass: {(end_time - start_time)*1000:.2f}ms")
        print(f"✅ Input Shape: {test_input.shape}")
        print(f"✅ Output Shape: {output.shape}")
        print(f"✅ Output Range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        return False

def main():
    """Run all GPU tests"""
    print("🎮 GPU Test Suite for UR3 Host System")
    print("=" * 60)
    
    results = {
        'gpu_available': test_gpu_availability(),
        'tensor_ops': test_tensor_operations(),
        'neural_network': test_neural_network()
    }
    
    print("\n📋 Test Summary")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if not result:
            all_passed = False
    
    print("\n🏁 Overall Result")
    print("=" * 50)
    if all_passed:
        print("✅ All tests passed! Your GPU setup is ready for UR3 deep learning.")
        print("💡 You can now run: python src/gpu_server.py")
    else:
        print("❌ Some tests failed. Please check your CUDA installation:")
        print("   1. Install NVIDIA GPU drivers")
        print("   2. Install CUDA toolkit (11.8+)")
        print("   3. Install PyTorch with CUDA support:")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    main()

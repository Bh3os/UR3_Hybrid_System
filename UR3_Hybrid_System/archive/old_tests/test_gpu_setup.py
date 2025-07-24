#!/usr/bin/env python3
"""
Test script to validate GPU server setup and functionality
Run this on the Windows host machine
"""

import sys
import time
import numpy as np
import socket
import json
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("Testing Python package imports...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('yaml', 'PyYAML'),
        ('PIL', 'Pillow'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name} - OK")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    else:
        print("\nAll required packages are installed!")
        return True

def test_cuda():
    """Test CUDA availability and GPU"""
    print("\nTesting CUDA and GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            print(f"✓ CUDA is available")
            print(f"✓ Number of GPU devices: {device_count}")
            print(f"✓ Current device: {current_device}")
            print(f"✓ Device name: {device_name}")
            
            # Test GPU memory
            total_memory = torch.cuda.get_device_properties(current_device).total_memory
            allocated_memory = torch.cuda.memory_allocated(current_device)
            
            print(f"✓ Total GPU memory: {total_memory / 1024**3:.2f} GB")
            print(f"✓ Allocated memory: {allocated_memory / 1024**3:.2f} GB")
            
            # Test tensor operations on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            
            print(f"✓ GPU tensor operations working")
            return True
            
        else:
            print("✗ CUDA is not available")
            print("Please check your CUDA installation and GPU drivers")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False

def test_model_loading():
    """Test model creation and loading"""
    print("\nTesting model creation...")
    
    try:
        # Import local modules
        sys.path.append('src')
        from enhanced_neural_network import UR3GraspCNN_Enhanced, create_model
        
        # Test model creation
        config = {
            'input_channels': 4,
            'input_size': [480, 640],
            'num_grasp_classes': 4,
            'output_6dof': True,
            'use_attention': True,
            'learning_rate': 1e-4
        }
        
        grasp_net, rl_module = create_model(config)
        print("✓ Model created successfully")
        
        # Test forward pass
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        dummy_input = torch.randn(1, 4, 480, 640).to(device)
        
        with torch.no_grad():
            outputs = grasp_net(dummy_input)
            
        print(f"✓ Forward pass successful")
        print(f"✓ Output shapes:")
        for key, value in outputs.items():
            print(f"  - {key}: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_network_server():
    """Test network server functionality"""
    print("\nTesting network server...")
    
    try:
        # Test socket creation
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Test binding to port
        test_port = 8888
        server_socket.bind(('localhost', test_port))
        server_socket.listen(1)
        
        print(f"✓ Server socket created and bound to port {test_port}")
        
        # Close socket
        server_socket.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Network server test failed: {e}")
        return False

def test_data_pipeline():
    """Test data processing pipeline"""
    print("\nTesting data pipeline...")
    
    try:
        import cv2
        import numpy as np
        
        # Create dummy RGBD image
        rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_image = np.random.uniform(0.5, 2.0, (480, 640)).astype(np.float32)
        
        print("✓ Dummy RGBD images created")
        
        # Test image processing
        sys.path.append('src')
        from data_pipeline import ImageProcessor
        
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processor = ImageProcessor(device)
        
        # Process images
        rgbd_tensor = processor.process_rgbd_image(rgb_image, depth_image)
        
        print(f"✓ RGBD processing successful, tensor shape: {rgbd_tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data pipeline test failed: {e}")
        return False

def test_gpu_server():
    """Test GPU server startup"""
    print("\nTesting GPU server startup...")
    
    try:
        sys.path.append('src')
        from gpu_server import GPUInferenceServer
        
        # Create server instance
        server = GPUInferenceServer()
        print("✓ GPU server created successfully")
        
        # Test server methods
        print(f"✓ Server device: {server.device}")
        print(f"✓ Model loaded: {server.model is not None}")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU server test failed: {e}")
        return False

def run_performance_benchmark():
    """Run performance benchmark"""
    print("\nRunning performance benchmark...")
    
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        sys.path.append('src')
        from enhanced_neural_network import UR3GraspCNN_Enhanced
        
        model = UR3GraspCNN_Enhanced().to(device)
        model.eval()
        
        # Benchmark inference time
        batch_sizes = [1, 4, 8, 16]
        num_iterations = 50
        
        print(f"Benchmarking on {device}...")
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 4, 480, 640).to(device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_iterations
            throughput = batch_size / avg_time
            
            print(f"  Batch size {batch_size:2d}: {avg_time*1000:6.2f} ms/batch, "
                  f"{throughput:6.1f} images/sec")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("UR3 Hybrid System - GPU Server Test Suite")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("CUDA and GPU", test_cuda),
        ("Model Loading", test_model_loading),
        ("Network Server", test_network_server),
        ("Data Pipeline", test_data_pipeline),
        ("GPU Server", test_gpu_server),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 20} {test_name} {'-' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your GPU server setup is ready.")
        print("\nNext steps:")
        print("1. Start the GPU server: python src/gpu_server.py")
        print("2. Configure VM to connect to this server")
        print("3. Run the complete hybrid system")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Update GPU drivers and CUDA")
        print("- Check firewall settings for network connectivity")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

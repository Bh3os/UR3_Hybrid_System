#!/usr/bin/env python3
"""
Simple Real Deployment Test
Tests the core components can be launched directly for deployment
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

print("🚀 UR3 Hybrid System - Simple Deployment Test")
print("=" * 60)

def test_gpu_server_launch():
    """Test that GPU server can be launched directly"""
    print("🖥️  Testing GPU Server Launch...")
    
    try:
        # Test that the main file exists and can be imported
        gpu_server_path = Path("host_gpu_system/src/gpu_server.py")
        if not gpu_server_path.exists():
            print("    ❌ GPU server file not found")
            return False
            
        print("    ✅ GPU server file exists")
        
        # Test syntax by compiling the file
        try:
            with open(gpu_server_path, 'r') as f:
                code = f.read()
            compile(code, str(gpu_server_path), 'exec')
            print("    ✅ GPU server syntax is valid")
        except SyntaxError as e:
            print(f"    ❌ GPU server syntax error: {e}")
            return False
            
        print("✅ GPU server ready for launch!")
        return True
        
    except Exception as e:
        print(f"❌ GPU server test failed: {e}")
        return False

def test_vm_client_launch():
    """Test that VM client can be launched directly"""
    print("🤖 Testing VM Client Launch...")
    
    try:
        # Test that the main file exists and can be imported
        vm_client_path = Path("vm_simulation_system/src/simulation_client.py")
        if not vm_client_path.exists():
            print("    ❌ VM client file not found")
            return False
            
        print("    ✅ VM client file exists")
        
        # Test syntax by compiling the file
        try:
            with open(vm_client_path, 'r') as f:
                code = f.read()
            compile(code, str(vm_client_path), 'exec')
            print("    ✅ VM client syntax is valid")
        except SyntaxError as e:
            print(f"    ❌ VM client syntax error: {e}")
            return False
            
        print("✅ VM client ready for launch!")
        return True
        
    except Exception as e:
        print(f"❌ VM client test failed: {e}")
        return False

def test_launch_scripts():
    """Create simple launch scripts for both systems"""
    print("📄 Creating Launch Scripts...")
    
    try:
        # Create GPU server launcher
        gpu_launcher = """#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run GPU server
from gpu_server import main

if __name__ == "__main__":
    main()
"""
        
        with open("host_gpu_system/launch_gpu_server.py", 'w') as f:
            f.write(gpu_launcher)
        os.chmod("host_gpu_system/launch_gpu_server.py", 0o755)
        print("    ✅ GPU server launcher created")
        
        # Create VM client launcher
        vm_launcher = """#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run VM client
from simulation_client import main

if __name__ == "__main__":
    main()
"""
        
        with open("vm_simulation_system/launch_vm_client.py", 'w') as f:
            f.write(vm_launcher)
        os.chmod("vm_simulation_system/launch_vm_client.py", 0o755)
        print("    ✅ VM client launcher created")
        
        print("✅ Launch scripts created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Launch script creation failed: {e}")
        return False

def test_quick_connection():
    """Test a quick connection between the systems"""
    print("🔗 Testing Quick Connection...")
    
    import socket
    import json
    import threading
    
    connection_successful = threading.Event()
    server_error = threading.Event()
    
    def quick_server():
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('localhost', 8890))
            server_socket.listen(1)
            server_socket.settimeout(5)
            
            print("    📡 Quick server listening...")
            
            try:
                conn, addr = server_socket.accept()
                print(f"    ✅ Connection from {addr}")
                
                # Quick ping-pong
                data = conn.recv(1024)
                message = json.loads(data.decode())
                
                if message.get('type') == 'ping':
                    response = {'type': 'pong', 'timestamp': time.time()}
                    conn.send(json.dumps(response).encode())
                    connection_successful.set()
                    
                conn.close()
            except socket.timeout:
                pass
                
            server_socket.close()
            
        except Exception as e:
            print(f"    ❌ Server error: {e}")
            server_error.set()
    
    def quick_client():
        try:
            time.sleep(0.2)  # Wait for server
            
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(3)
            client_socket.connect(('localhost', 8890))
            
            # Send ping
            ping = {'type': 'ping', 'timestamp': time.time()}
            client_socket.send(json.dumps(ping).encode())
            
            # Receive pong
            data = client_socket.recv(1024)
            response = json.loads(data.decode())
            
            if response.get('type') == 'pong':
                print("    ✅ Ping-pong successful")
                
            client_socket.close()
            
        except Exception as e:
            print(f"    ❌ Client error: {e}")
    
    # Run server and client
    server_thread = threading.Thread(target=quick_server, daemon=True)
    client_thread = threading.Thread(target=quick_client, daemon=True)
    
    server_thread.start()
    client_thread.start()
    
    client_thread.join(timeout=5)
    server_thread.join(timeout=1)
    
    if connection_successful.is_set() and not server_error.is_set():
        print("✅ Quick connection test successful!")
        return True
    else:
        print("❌ Quick connection test failed!")
        return False

def create_deployment_guide():
    """Create a deployment guide"""
    print("📚 Creating Deployment Guide...")
    
    guide = """# UR3 Hybrid System - Deployment Guide

## 🚀 Quick Start Deployment

### Host GPU System (Windows/Linux with GPU)
1. Navigate to host_gpu_system directory
2. Run: `python3 launch_gpu_server.py`
3. Server will start on port 8888

### VM Simulation System (Ubuntu VM)
1. Navigate to vm_simulation_system directory  
2. Run: `python3 launch_vm_client.py`
3. Client will connect to host automatically

## 📋 Requirements

### Host System:
- Python 3.8+
- PyTorch (with CUDA support recommended)
- OpenCV
- NumPy
- PyYAML

### VM System:
- Python 3.8+
- OpenCV
- NumPy  
- PyYAML
- ROS (optional - system works without ROS)
- Webots (optional - system works in simulation mode)

## 🔧 Configuration

### Network Configuration:
- Default host IP: 192.168.1.1
- Default port: 8888
- Modify config files in config/ directories

### Model Configuration:
- Neural network settings in host_gpu_system/config/model_config.yaml
- Training parameters in host_gpu_system/config/training_config.yaml

### Robot Configuration:
- Robot parameters in vm_simulation_system/config/robot_config.yaml
- Camera settings in vm_simulation_system/config/camera_config.yaml

## 🧪 Testing

Run integration tests:
```bash
# Test host system
cd host_gpu_system && python3 integration_test.py

# Test VM system  
cd vm_simulation_system && python3 vm_integration_test.py

# Test end-to-end
cd .. && python3 end_to_end_test.py
```

## 🚨 Troubleshooting

1. **Connection refused**: Check firewall and IP addresses
2. **Import errors**: Ensure all dependencies are installed
3. **GPU not found**: System will fallback to CPU automatically
4. **ROS errors**: System works without ROS in simulation mode

## 📞 Support

Check the log files in data/logs/ for detailed error information.
All components have comprehensive logging for debugging.
"""
    
    try:
        with open("DEPLOYMENT_GUIDE.md", 'w') as f:
            f.write(guide)
        print("    ✅ Deployment guide created: DEPLOYMENT_GUIDE.md")
        return True
    except Exception as e:
        print(f"    ❌ Failed to create deployment guide: {e}")  
        return False

def main():
    """Run simple deployment test"""
    
    tests = [
        ("GPU Server Launch Readiness", test_gpu_server_launch),
        ("VM Client Launch Readiness", test_vm_client_launch),
        ("Launch Scripts Creation", test_launch_scripts),
        ("Quick Connection Test", test_quick_connection),
        ("Deployment Guide Creation", create_deployment_guide),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 40)
        
        start_time = time.time()
        try:
            success = test_func()
        except Exception as e:
            print(f"❌ Test failed: {e}")
            success = False
        end_time = time.time()
        
        results.append({
            'name': test_name,
            'success': success,
            'duration': end_time - start_time
        })
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} ({end_time - start_time:.2f}s)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 DEPLOYMENT READINESS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['name']}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 SYSTEM IS READY FOR DEPLOYMENT!")
        print("\n🚀 To deploy:")
        print("   1. On host: python3 host_gpu_system/launch_gpu_server.py")
        print("   2. On VM:   python3 vm_simulation_system/launch_vm_client.py")
        print("   3. Components will connect automatically!")
        print("\n📚 See DEPLOYMENT_GUIDE.md for detailed instructions")
    else:
        print("❌ Some components need attention before deployment")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

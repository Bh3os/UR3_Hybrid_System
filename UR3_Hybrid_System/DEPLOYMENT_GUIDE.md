# UR3 Hybrid System - Deployment Guide

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

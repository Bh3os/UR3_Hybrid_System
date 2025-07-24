# Host GPU System Documentation

## 🖥️ Overview
This module runs on your **Windows host machine** with the **RTX A6000 GPU** and handles all deep learning computations for the UR3 grasping system.

## 🏗️ Architecture

```
Host GPU System (Windows)
├── GPU Server ────────────── Main communication server
├── Neural Network ────────── CNN for grasp prediction  
├── Training Module ───────── Deep RL training loop
├── Data Pipeline ─────────── Image processing & batching
└── Configuration ─────────── Network & model settings
```

## 📁 Directory Structure

```
host_gpu_system/
├── README.md              # This documentation
├── requirements.txt       # Python dependencies
├── setup.bat             # Windows setup script
├── gpu_test.py           # GPU verification script
│
├── src/                  # Source code
│   ├── gpu_server.py     # 🌟 Main GPU server
│   ├── neural_network.py # CNN architecture
│   ├── training.py       # RL training loop
│   ├── data_pipeline.py  # Data processing
│   └── utils/
│       ├── logger.py     # Logging utilities
│       ├── metrics.py    # Performance metrics
│       └── visualization.py # Result plotting
│
├── config/               # Configuration files
│   ├── network_config.yaml  # Network settings
│   ├── model_config.yaml    # Neural network config
│   └── training_config.yaml # Training parameters
│
├── models/               # Model storage
│   ├── ur3_model.pth    # Trained weights
│   ├── checkpoints/     # Training checkpoints
│   └── exports/         # ONNX/TensorRT models
│
└── data/                # Data storage
    ├── training_images/ # Training dataset
    ├── validation/      # Validation data
    └── logs/           # Training logs
```

## ⚙️ Module Components

### 1. GPU Server (`src/gpu_server.py`)
**Purpose**: Main communication hub receiving data from VM and returning predictions

**Key Functions**:
- `start_server()`: Initialize TCP server on port 8888
- `handle_vm_connection()`: Process incoming VM requests
- `predict_grasp()`: Run neural network inference
- `preprocess_image()`: Convert VM data to tensor format

**Workflow**:
```
1. Listen for VM connections on port 8888
2. Receive RGBD camera data from VM
3. Preprocess images (normalize, tensorize)
4. Run GPU inference using trained CNN
5. Return 6-DOF grasp pose to VM
6. Log performance metrics
```

### 2. Neural Network (`src/neural_network.py`)
**Purpose**: CNN architecture for processing RGBD images and predicting grasp poses

**Architecture**:
```python
Input: RGBD Image (4 channels, 480x640)
├── Conv2D Layers (32, 64, 128 filters)
├── Max Pooling & ReLU activations
├── Adaptive Global Average Pooling
├── Fully Connected Layers (512, 256 neurons)
└── Output: 6-DOF Pose (x, y, z, rx, ry, rz)
```

**Key Features**:
- CUDA-optimized for RTX A6000
- Batch processing support
- Model checkpointing
- Transfer learning capabilities

### 3. Training Module (`src/training.py`)
**Purpose**: Deep reinforcement learning training loop

**Components**:
- Experience replay buffer
- Q-learning with neural network approximation
- Epsilon-greedy exploration
- Target network updates
- Performance monitoring

### 4. Data Pipeline (`src/data_pipeline.py`)
**Purpose**: Efficient data handling and preprocessing

**Features**:
- Image compression/decompression
- Batch processing
- Data augmentation
- Memory management
- CUDA tensor operations

## 🚀 Installation & Setup

### Prerequisites
- Windows 10/11 (64-bit)
- NVIDIA RTX A6000 GPU
- Python 3.8-3.11
- CUDA 11.8+
- 16GB+ available RAM

### Step 1: Environment Setup
```powershell
# Clone or extract the project
cd host_gpu_system

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
setup.bat
```

### Step 2: GPU Verification
```powershell
# Test GPU availability
python gpu_test.py

# Expected output:
# ✅ CUDA Available: True
# ✅ GPU: NVIDIA RTX A6000
# ✅ VRAM: 48.0GB available
```

### Step 3: Configuration
Edit configuration files in `config/`:

**network_config.yaml**:
```yaml
network:
  host_ip: "0.0.0.0"      # Listen on all interfaces
  port: 8888              # Communication port
  vm_ip: "192.168.1.100"  # VM IP address (update this!)
  timeout: 30             # Connection timeout
  buffer_size: 4096       # Socket buffer size
```

**model_config.yaml**:
```yaml
model:
  input_channels: 4       # RGBD input
  image_size: [480, 640]  # Camera resolution
  output_dim: 6           # 6-DOF pose
  batch_size: 16          # Inference batch size
  device: "cuda:0"        # GPU device
```

## 🏃‍♂️ Running the System

### Start GPU Server
```powershell
# Navigate to host system
cd host_gpu_system

# Activate environment
venv\Scripts\activate

# Start the main server
python src/gpu_server.py

# Expected output:
# 🚀 GPU Server initialized on cuda:0
# 📡 Listening for VM connection on 192.168.1.100:8888
# 🔄 Server started, waiting for VM connections...
```

### Monitor Performance
```powershell
# Check GPU utilization
nvidia-smi

# View training logs
python src/utils/monitor.py

# Visualize results
python src/utils/visualization.py
```

## 🔧 Configuration Options

### Network Settings
- **Port**: Default 8888 (ensure firewall allows)
- **IP Address**: Use VM's bridged network IP
- **Timeout**: Adjust based on network latency
- **Buffer Size**: Optimize for image transfer

### Model Settings
- **Batch Size**: Increase for better GPU utilization
- **Learning Rate**: Adjust for training stability
- **Architecture**: Modify layers in neural_network.py
- **Device**: Use "cuda:0" for RTX A6000

### Performance Tuning
- **Memory Management**: Enable CUDA memory caching
- **Mixed Precision**: Use FP16 for faster inference
- **Model Optimization**: Export to TensorRT for production
- **Batch Processing**: Process multiple images together

## 📊 Performance Monitoring

### Key Metrics
- **Inference Speed**: Target 50-100 FPS
- **GPU Utilization**: Monitor with nvidia-smi
- **Memory Usage**: Track VRAM consumption
- **Network Latency**: Measure round-trip time

### Logging
All operations are logged to `data/logs/`:
- `gpu_server.log`: Server operations
- `training.log`: Training progress
- `performance.log`: Performance metrics
- `errors.log`: Error tracking

## 🐛 Troubleshooting

### Common Issues

**"CUDA Out of Memory"**:
```python
# Reduce batch size in config/model_config.yaml
batch_size: 8  # Instead of 16

# Clear GPU cache
torch.cuda.empty_cache()
```

**"Connection Refused from VM"**:
```powershell
# Check Windows Firewall
netsh advfirewall firewall add rule name="UR3_GPU_Server" dir=in action=allow protocol=TCP localport=8888

# Verify IP configuration
ipconfig
```

**Slow Inference Speed**:
```python
# Enable optimizations in neural_network.py
model = torch.jit.script(model)  # JIT compilation
model.half()  # FP16 precision
```

### Performance Optimization

1. **GPU Memory**: Monitor VRAM usage and optimize batch sizes
2. **CPU Usage**: Use multiple workers for data loading
3. **Network**: Use image compression for faster transfer
4. **Storage**: Use NVMe SSD for model/data storage

## 🔗 Integration Points

### VM Communication
- **Input**: RGBD images from VM simulation
- **Output**: 6-DOF grasp poses for robot control
- **Protocol**: TCP socket with JSON messaging
- **Data Format**: See `shared_resources/data_formats.py`

### Training Data Flow
```
VM Simulation → Host GPU → Neural Network → Prediction → VM Robot
     ↑                                           ↓
Training Data ← Performance Metrics ← Execution Results
```

## 📈 Expected Performance

With RTX A6000 and optimized settings:
- **Inference Speed**: 80-120 FPS
- **Training Speed**: 1000-2000 episodes/hour
- **Memory Usage**: 8-16GB VRAM
- **Network Latency**: 1-3ms to VM

## 🔄 Maintenance

### Regular Tasks
- Monitor GPU temperature and usage
- Clean up old log files
- Update model checkpoints
- Backup training data

### Updates
- Check for CUDA driver updates
- Update PyTorch for performance improvements
- Monitor for dependency security patches

---

**🎯 This host system leverages your RTX A6000's full power for optimal deep learning performance while maintaining seamless communication with the VM simulation environment.**

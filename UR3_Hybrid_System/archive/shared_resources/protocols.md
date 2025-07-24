# Communication Protocols Documentation

## 🌐 Host-VM Communication Protocols

This document describes the communication protocols used between the Windows host (GPU server) and Ubuntu VM (simulation client).

## 📡 Network Architecture

```
┌─────────────────────────┐         ┌─────────────────────────┐
│    Windows Host         │  TCP    │      Ubuntu VM          │
│   (GPU Server)          │◄────────┤  (Simulation Client)    │
│   192.168.1.1:8888     │ Socket  │   192.168.1.100         │
└─────────────────────────┘         └─────────────────────────┘
```

## 🔌 Connection Protocol

### 1. Connection Establishment
- **Protocol**: TCP Socket
- **Port**: 8888 (configurable)
- **Timeout**: 30 seconds
- **Retry Logic**: 5 attempts with exponential backoff

### 2. Message Format
All messages use JSON encoding with length prefix:
```
[4-byte length][JSON message]
```

**Length Encoding**: Big-endian 32-bit integer
**Message Encoding**: UTF-8 JSON

## 📨 Message Types

### 1. Camera Data Message (VM → Host)
```json
{
  "type": "camera_data",
  "data": {
    "rgb": [[R, G, B], ...],          // RGB image as nested list (480x640x3)
    "depth": [depth_values, ...],     // Depth image as flat list (480x640)
    "timestamp": 1642534567.123,
    "episode": 42
  },
  "robot_state": {
    "names": ["joint1", "joint2", ...],
    "positions": [0.1, -0.5, ...],
    "velocities": [0.0, 0.0, ...],
    "efforts": [0.0, 0.0, ...]
  }
}
```

### 2. Grasp Prediction Response (Host → VM)
```json
{
  "type": "grasp_prediction",
  "pose": [x, y, z, rx, ry, rz],      // 6-DOF grasp pose
  "confidence": 0.85,                  // Prediction confidence (0-1)
  "timestamp": 1642534567.125,
  "processing_time": 0.023             // Inference time in seconds
}
```

### 3. Execution Feedback (VM → Host)
```json
{
  "type": "execution_feedback",
  "success": true,                     // Grasp execution success
  "pose": [x, y, z, rx, ry, rz],      // Executed pose
  "actual_joint_positions": [...],     // Final joint positions
  "execution_time": 2.5,              // Time to execute (seconds)
  "timestamp": 1642534569.650,
  "episode": 42
}
```

### 4. Episode Management

#### Episode Start (VM → Host)
```json
{
  "type": "episode_start",
  "episode": 43,
  "timestamp": 1642534570.000,
  "environment_config": {
    "objects": ["cube", "sphere"],
    "table_color": [0.8, 0.6, 0.4]
  }
}
```

#### Episode End (VM → Host)
```json
{
  "type": "episode_end",
  "episode": 43,
  "duration": 45.2,
  "success": true,
  "actions_count": 15,
  "total_reward": 8.5,
  "timestamp": 1642534615.200
}
```

### 5. Training Data (VM → Host)
```json
{
  "type": "training_data",
  "episode": 43,
  "state": {
    "rgb": [...],
    "depth": [...],
    "robot_joints": [...]
  },
  "action": [x, y, z, rx, ry, rz],
  "reward": 1.0,
  "next_state": {...},
  "done": false,
  "timestamp": 1642534575.123
}
```

### 6. System Control Messages

#### Ping/Pong (Heartbeat)
```json
// VM → Host
{
  "type": "ping",
  "timestamp": 1642534567.123
}

// Host → VM
{
  "type": "pong",
  "timestamp": 1642534567.125
}
```

#### Error Response
```json
{
  "type": "error",
  "message": "Invalid message format",
  "error_code": "INVALID_FORMAT",
  "timestamp": 1642534567.123
}
```

## 🔄 Communication Flow

### 1. Normal Operation Flow
```
VM                          Host
│                           │
├─ Connect ─────────────────►│
│◄─ Accept ──────────────────┤
│                           │
├─ camera_data ─────────────►│
│                           ├─ Process with GPU
│◄─ grasp_prediction ───────┤
│                           │
├─ Execute grasp           │
├─ execution_feedback ─────►│
│                           ├─ Update model
│                           │
└─ Repeat ─────────────────►│
```

### 2. Episode Management Flow
```
VM                          Host
│                           │
├─ episode_start ──────────►│
│                           ├─ Initialize episode
│                           │
├─ Multiple camera_data ───►│
│◄─ Multiple predictions ───┤
│                           │
├─ episode_end ────────────►│
│                           ├─ Process episode data
│                           ├─ Update training
│                           │
└─ Next episode ───────────►│
```

## ⚙️ Configuration Parameters

### Network Settings
```yaml
network:
  host_ip: "192.168.1.1"        # Windows host IP
  host_port: 8888               # Communication port
  vm_ip: "192.168.1.100"        # VM IP
  timeout: 30                   # Connection timeout (seconds)
  retry_attempts: 5             # Connection retry count
  buffer_size: 4096             # Socket buffer size
  heartbeat_interval: 10        # Ping interval (seconds)
```

### Data Compression
```yaml
compression:
  enabled: true                 # Enable image compression
  quality: 80                   # JPEG quality (0-100)
  depth_scaling: 1000           # Depth value scaling factor
```

## 🚀 Performance Optimization

### 1. Image Compression
```python
# VM side - compress before sending
import cv2
import base64

def compress_image(image, quality=80):
    _, buffer = cv2.imencode('.jpg', image, 
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode()

# Host side - decompress after receiving
def decompress_image(compressed_data):
    buffer = base64.b64decode(compressed_data)
    nparr = np.frombuffer(buffer, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
```

### 2. Batch Processing
```python
# Process multiple images together
def process_image_batch(images):
    batch = torch.stack([preprocess(img) for img in images])
    with torch.no_grad():
        predictions = model(batch)
    return predictions.cpu().numpy()
```

### 3. Connection Pooling
```python
# Maintain persistent connections
class ConnectionPool:
    def __init__(self, max_connections=5):
        self.connections = []
        self.max_connections = max_connections
    
    def get_connection(self):
        # Return available connection or create new one
        pass
```

## 🛡️ Error Handling

### 1. Connection Errors
- **Timeout**: Retry with exponential backoff
- **Refused**: Check host availability
- **Network**: Switch to backup communication method

### 2. Message Errors
- **Invalid JSON**: Send error response with details
- **Missing Fields**: Use default values where possible
- **Type Mismatch**: Convert types when safe

### 3. Recovery Strategies
```python
def handle_communication_error(error_type):
    if error_type == "connection_lost":
        return reconnect_with_backoff()
    elif error_type == "message_corrupted":
        return request_retransmission()
    elif error_type == "timeout":
        return reduce_message_size()
```

## 📊 Monitoring & Debugging

### 1. Performance Metrics
- **Latency**: Round-trip message time
- **Throughput**: Messages per second
- **Error Rate**: Failed message percentage
- **Connection Uptime**: Connection stability

### 2. Debug Messages
```json
{
  "type": "debug_info",
  "latency_ms": 15.2,
  "queue_size": 3,
  "memory_usage": "2.4GB",
  "gpu_utilization": "85%",
  "timestamp": 1642534567.123
}
```

### 3. Logging Format
```
[TIMESTAMP] [LEVEL] [COMPONENT] Message
2024-01-18 14:30:15.123 INFO  HOST_SERVER Received camera data from VM
2024-01-18 14:30:15.145 DEBUG GPU_MODEL   Inference completed in 22ms
2024-01-18 14:30:15.147 INFO  HOST_SERVER Sent grasp prediction to VM
```

## 🔧 Troubleshooting Guide

### Common Issues
1. **"Connection Refused"**: Check firewall, IP addresses, port availability
2. **"Message Too Large"**: Implement image compression or reduce resolution
3. **"JSON Decode Error"**: Verify message format and encoding
4. **"Timeout"**: Increase timeout values or optimize processing speed

### Testing Tools
```bash
# Test connection
nc -zv 192.168.1.1 8888

# Monitor network traffic
tcpdump -i any port 8888

# Test JSON parsing
echo '{"type":"ping"}' | python3 -m json.tool
```

---

**📡 This protocol ensures reliable, efficient communication between your Windows host and Ubuntu VM for optimal UR3 system performance.**

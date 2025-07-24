# Webots Launch Configuration for UR3 Hybrid System

This configuration launches the complete Webots simulation environment with the UR3 robot, camera, and blocks.

## Files Required

### Webots World File
- **Location**: `Webots/worlds/Environment.wbt`
- **Description**: Main simulation environment with UR3 robot, Kinect camera, and manipulable blocks
- **Features**:
  - UR3e robot with gripper
  - Kinect RGB-D camera
  - 5 manipulable blocks with ArUco markers
  - Realistic lighting and physics

### Robot Prototypes
- **Location**: `Webots/protos/`
- **Files**:
  - `UR3e.proto` - UR3e robot model
  - `Kinect.proto` - Kinect camera model
  - `KukaGripper.proto` - Gripper model
  - `Robotiq85Gripper.proto` - Alternative gripper

### Textures and Assets
- **Location**: `Webots/protos/textures/` and `Webots/protos/icons/`
- **Purpose**: Visual assets for realistic rendering

## Launch Instructions

### Option 1: Full Webots Mode (Recommended for Full Simulation)

1. **Install Webots** (if not already installed):
   ```bash
   # Download from https://cyberbotics.com/
   # Or install via package manager
   sudo apt install webots  # Ubuntu
   brew install webots      # macOS
   ```

2. **Launch Webots with our world**:
   ```bash
   cd vm_simulation_system
   webots Webots/worlds/Environment.wbt
   ```

3. **Start simulation controllers**:
   ```bash
   # Terminal 1: Start ROS core
   roscore
   
   # Terminal 2: Start VM simulation client
   python3 src/simulation_client.py --webots-mode
   ```

### Option 2: Headless Mode (For Remote/Server Deployment)

1. **Launch headless Webots**:
   ```bash
   webots --batch --mode=fast Webots/worlds/Environment.wbt
   ```

2. **Start simulation client**:
   ```bash
   python3 src/simulation_client.py --webots-headless
   ```

### Option 3: Mock Mode (For Development/Testing)

1. **Launch without Webots**:
   ```bash
   python3 src/simulation_client.py --simulation-mode
   ```

## Simulation Features

### Robot Control
- **6-DOF UR3e robot arm**
- **Forward/inverse kinematics**
- **Joint position control**
- **End-effector pose control**
- **Gripper control**

### Vision System
- **RGB camera (640x480)**
- **Depth camera**
- **Real-time image streaming**
- **ROS image transport**

### Environment
- **5 manipulable blocks**
- **Block pose tracking**
- **Physics simulation**
- **Collision detection**

### Data Flow
```
Webots Simulation
    ↓
WebotsBridge (supervisor + camera)
    ↓
SimulationClient
    ↓
Socket Communication
    ↓
Host GPU System (RL Training)
```

## Configuration

### Robot Configuration
Edit `config/robot_config.yaml`:
```yaml
robot:
  type: "ur3e"
  base_position: [0.69, 0.74, 0]
  joint_limits:
    - [-6.28, 6.28]  # Joint 1
    - [-6.28, 6.28]  # Joint 2
    # ... etc
```

### Camera Configuration  
Edit `config/camera_config.yaml`:
```yaml
camera:
  type: "kinect"
  resolution: [640, 480]
  framerate: 30
  topics:
    rgb: "/camera/rgb/image_raw"
    depth: "/camera/depth/image_raw"
```

### Webots Configuration
Edit `config/webots_config.yaml`:
```yaml
webots:
  world_file: "Environment.wbt"
  timestep: 32
  physics_timestep: 16
  supervisor_name: "supervisor"
  camera_name: "camera"
  blocks:
    count: 5
    prefix: "block"
```

## Troubleshooting

### "Webots Not Found"
```bash
# Check Webots installation
which webots

# Add to PATH if needed
export PATH=$PATH:/Applications/Webots.app
```

### "Controller Connection Failed"
```bash
# Ensure controllers are named correctly in Webots
# Check robot names in world file match config
```

### "ROS Topics Not Publishing"
```bash
# Check ROS master
rostopic list

# Verify camera topics
rostopic echo /camera/rgb/image_raw
```

### "Simulation Running Slowly"
```bash
# Reduce physics timestep in world file
# Set lower camera framerate
# Use fast mode: webots --mode=fast
```

## Integration with UR3 Hybrid System

The Webots simulation integrates seamlessly with the UR3 Hybrid System:

1. **Webots provides**:
   - Realistic physics simulation
   - Visual rendering
   - Sensor data (RGB-D images)
   - Robot control interface

2. **VM Simulation System provides**:
   - High-level robot control
   - Image processing
   - Data communication to GPU server

3. **Host GPU System provides**:
   - Deep learning inference
   - RL training
   - Action planning

This creates a complete pipeline from simulation to AI training and back to robot control.

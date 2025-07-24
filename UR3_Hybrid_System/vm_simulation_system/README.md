# VM Simulation System Documentation

##   Overview
This module runs on your **Ubuntu 18.04 VM** and handles the Webots simulation, ROS integration, and robot control for the UR3 grasping system.

##  Architecture

```
VM Simulation System (Ubuntu 18.04 + ROS Melodic)
├── Simulation Client ────────── Main communication client
├── Robot Controller ─────────── UR3 robot control & kinematics
├── Camera Handler ───────────── RGBD camera processing
├── Webots Integration ───────── Physics simulation interface
└── ROS Node Manager ─────────── ROS topics & services
```

##  Directory Structure

```
vm_simulation_system/
├── README.md              # This documentation
├── requirements.txt       # Ubuntu/ROS dependencies
├── setup.sh              # Ubuntu setup script
├── package.xml           # ROS package configuration
│
├── src/                  # Source code
│   ├── simulation_client.py   #  Main VM client
│   ├── enhanced_robot_controller.py    # Enhanced UR3 robot control with full kinematics
│   ├── enhanced_camera_handler.py      # Advanced camera data processing and vision
│   ├── webots_bridge.py       # Webots integration
│   └── integrator/            # Original integrator modules
│       ├── __init__.py
│       ├── camera.py          # Camera interface
│       ├── gripper_user.py    # Gripper control
│       ├── Kinematics.py      # UR3 kinematics
│       ├── supervisor.py      # Webots supervisor
│       └── watchdog.py        # System monitoring
│
├── launch/               # ROS launch files
│   ├── ur3_simulation.launch  # Main simulation launcher
│   ├── webots_bridge.launch   # Webots ROS bridge
│   └── camera_nodes.launch    # Camera processing nodes
│
├── config/               # Configuration files
│   ├── ros_config.yaml        # ROS parameters
│   ├── robot_config.yaml      # UR3 robot settings
│   └── network_config.yaml    # Host communication
│
├── webots_worlds/        # Webots simulation files
│   ├── ur3_environment.wbt    # Main world file
│   ├── protos/               # Robot prototypes
│   └── textures/             # Environment textures
│
└── msg/                  # ROS message definitions
    ├── RobotState.msg        # Robot state message
    └── GraspCommand.msg      # Grasp command message
```

##  Module Components

### 1. Simulation Client (`src/simulation_client.py`)
**Purpose**: Main communication hub connecting to host GPU server

**Key Functions**:
- `connect_to_host()`: Establish TCP connection to Windows host
- `send_camera_data()`: Send RGBD images to host for processing
- `execute_grasp_action()`: Execute predicted grasp in simulation
- `manage_episodes()`: Handle training episode lifecycle

**Workflow**:
```
1. Initialize ROS node and subscribers
2. Connect to host GPU server
3. Receive camera data from Webots
4. Send data to host for neural network processing
5. Receive grasp predictions from host
6. Execute grasp actions through robot controller
7. Send feedback and rewards back to host
```

### 2. Enhanced Robot Controller (`src/enhanced_robot_controller.py`)
**Purpose**: Advanced UR3 robot control with full kinematics, trajectory planning, and safety

**Key Features**:
- Complete forward and inverse kinematics with DH parameters
- Advanced joint trajectory planning with cubic splines
- Real-time collision avoidance and safety monitoring
- Precise grasp execution with force feedback
- Multi-modal control (position, velocity, torque)
- Comprehensive workspace analysis

### 3. Enhanced Camera Handler (`src/enhanced_camera_handler.py`)
**Purpose**: Advanced vision processing for RGB-D data with AI-enhanced features

**Functions**:
- Multi-camera RGB-D image processing and alignment
- Advanced depth filtering with bilateral and temporal filters
- Real-time object detection and segmentation
- Camera calibration and distortion correction
- Intelligent data compression and streaming
- Vision-based workspace monitoring

### 4. Webots Bridge (`src/webots_bridge.py`)
**Purpose**: Interface between ROS and Webots simulation

**Capabilities**:
- Sensor data extraction
- Robot command execution
- Environment manipulation
- Physics parameter control
- Scene randomization

### 5. Integrator Modules (`src/integrator/`)
**Purpose**: Enhanced versions of your original integrator code

**Components**:
- `camera.py`: Intel RealSense simulation
- `gripper_user.py`: Gripper control and sensing
- `Kinematics.py`: UR3 mathematical models
- `supervisor.py`: Environment supervision
- `watchdog.py`: System health monitoring

## Installation & Setup

### Prerequisites
- Ubuntu 18.04 LTS (VM with 32GB RAM, 16 CPU cores)
- ROS Melodic Desktop Full
- Webots R2023a
- Python 3.6+
- Network access to Windows host

### Step 1: VM Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y build-essential git vim curl wget

# Install ROS Melodic (if not already installed)
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install -y ros-melodic-desktop-full

# Initialize rosdep
sudo rosdep init
rosdep update
```

### Step 2: Environment Setup
```bash
# Navigate to VM system directory
cd vm_simulation_system

# Make setup script executable
chmod +x setup.sh

# Run setup script
./setup.sh

# Source ROS environment
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Webots Installation
```bash
# Download Webots
wget -O webots.tar.bz2 "https://github.com/cyberbotics/webots/releases/download/R2023a/webots-R2023a-x86-64.tar.bz2"

# Extract and install
tar -xjf webots.tar.bz2
sudo mv webots /opt/
sudo ln -sf /opt/webots/webots /usr/local/bin/webots

# Set environment variables
echo 'export WEBOTS_HOME=/opt/webots' >> ~/.bashrc
echo 'export PATH=$PATH:$WEBOTS_HOME' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Configuration
Edit configuration files in `config/`:

**network_config.yaml**:
```yaml
network:
  host_ip: "192.168.1.1"     # Windows host IP (update this!)
  host_port: 8888            # GPU server port
  vm_ip: "192.168.1.100"     # This VM's IP
  timeout: 30                # Connection timeout
  retry_attempts: 5          # Connection retry count
```

**robot_config.yaml**:
```yaml
robot:
  model: "UR3e"
  dof: 6
  joint_limits:
    - [-3.14159, 3.14159]    # Joint 1
    - [-3.14159, 3.14159]    # Joint 2
    - [-3.14159, 3.14159]    # Joint 3
    - [-3.14159, 3.14159]    # Joint 4
    - [-3.14159, 3.14159]    # Joint 5
    - [-3.14159, 3.14159]    # Joint 6
  
camera:
  resolution: [640, 480]
  frame_rate: 30
  depth_range: [0.1, 2.0]
```

## Running the System

### Start ROS Core and Simulation
```bash
# Terminal 1: Start ROS core
roscore

# Terminal 2: Launch Webots simulation
roslaunch vm_simulation_system ur3_simulation.launch

# Terminal 3: Start camera processing
roslaunch vm_simulation_system camera_nodes.launch

# Terminal 4: Start main simulation client
cd vm_simulation_system
python3 src/simulation_client.py
```

### Single Command Launch
```bash
# Launch everything together
roslaunch vm_simulation_system full_system.launch host_ip:=192.168.1.1
```

## ROS Topics and Services

### Published Topics
- `/camera/image_raw` (sensor_msgs/Image): RGB camera data
- `/camera/depth/image_raw` (sensor_msgs/Image): Depth camera data
- `/ur3/joint_states` (sensor_msgs/JointState): Robot joint positions
- `/gripper/state` (std_msgs/Bool): Gripper open/closed state

### Subscribed Topics
- `/ur3/joint_commands` (trajectory_msgs/JointTrajectory): Joint commands
- `/gripper/command` (std_msgs/Bool): Gripper control
- `/simulation/reset` (std_msgs/Empty): Environment reset

### Services
- `/ur3/get_pose` (geometry_msgs/PoseStamped): Get current end-effector pose
- `/simulation/randomize` (std_srvs/Empty): Randomize environment
- `/gripper/grasp` (std_srvs/SetBool): Execute grasp action

## Configuration Options

### Network Settings
- **Host IP**: Windows machine IP address
- **Connection Timeout**: Adjust based on network conditions
- **Retry Logic**: Configure reconnection attempts
- **Buffer Sizes**: Optimize for image data transfer

### Robot Parameters
- **Joint Limits**: Safety constraints for robot motion
- **Velocity Limits**: Maximum joint velocities
- **Acceleration Limits**: Maximum joint accelerations
- **Workspace Bounds**: Reachable workspace definition

### Simulation Settings
- **Physics Timestep**: Webots simulation accuracy vs speed
- **Camera Settings**: Resolution and frame rate
- **Environment Randomization**: Object placement variety
- **Reward Function**: Training feedback parameters

## Troubleshooting

### Common Issues

**"Cannot connect to host" Error**:
```bash
# Check network connectivity
ping 192.168.1.1  # Host IP

# Verify VM IP configuration
ip addr show

# Test port connectivity
nc -zv 192.168.1.1 8888
```

**ROS Node Communication Issues**:
```bash
# Check ROS master
echo $ROS_MASTER_URI

# List active topics
rostopic list

# Monitor topic data
rostopic echo /camera/image_raw
```

**Webots Startup Problems**:
```bash
# Check Webots installation
webots --version

# Verify environment variables
echo $WEBOTS_HOME

# Test with simple world
webots /opt/webots/projects/samples/tutorials/worlds/tutorial1.wbt
```

**Python Import Errors**:
```bash
# Check Python path
python3 -c "import sys; print(sys.path)"

# Verify ROS Python packages
python3 -c "import rospy; print('ROS OK')"

# Install missing packages
pip3 install -r requirements.txt
```

### Performance Tuning

1. **VM Resources**: Allocate more CPU cores and RAM if available
2. **Graphics**: Enable 3D acceleration in VM settings  
3. **Network**: Use bridged networking for lower latency
4. **Storage**: Use SSD storage for VM disk files

## Integration Points

### Host Communication
- **Data Sent**: RGBD camera images, robot state, episode information
- **Data Received**: Grasp predictions, training commands, configuration updates
- **Protocol**: TCP socket with JSON messaging
- **Error Handling**: Automatic reconnection and data buffering

### Webots Integration
```python
# Example Webots controller integration
from controller import Robot, Camera, Motor

robot = Robot()
camera = robot.getDevice("camera")
camera.enable(timestep)

# Get image data
image = camera.getImage()
# Process and send to host...
```

## Expected Performance

With proper VM configuration:
- **Simulation Speed**: Real-time or faster
- **Camera Frame Rate**: 30 FPS
- **Network Latency**: 1-5ms to host
- **Episode Duration**: 30-60 seconds
- **Training Episodes**: 100+ per hour

## Maintenance

### Regular Tasks
- Monitor VM resource usage
- Check ROS log files for errors  
- Update Webots worlds and models
- Backup simulation data and configs

### Debugging Tools
```bash
# ROS debugging
rosrun rqt_console rqt_console     # View ROS logs
rosrun rqt_graph rqt_graph         # Visualize node graph
rosrun rviz rviz                   # 3D visualization

# System monitoring
htop                               # CPU/Memory usage
nvidia-smi                         # GPU usage (if available)
iftop                             # Network traffic
```

---

**This VM system provides the complete ROS/Webots simulation environment while seamlessly communicating with your Windows host for GPU-accelerated deep learning!**

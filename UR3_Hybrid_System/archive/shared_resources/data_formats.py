#!/usr/bin/env python3
"""
Shared Data Structures for UR3 Hybrid System
Common data formats used between host and VM components
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import json
import time

@dataclass
class RGBDImage:
    """RGBD image data structure"""
    rgb: np.ndarray          # RGB image (H, W, 3)
    depth: np.ndarray        # Depth image (H, W)
    timestamp: float         # Capture timestamp
    frame_id: int            # Frame identifier
    camera_info: Dict        # Camera calibration info
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'rgb': self.rgb.tolist(),
            'depth': self.depth.tolist(),
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'camera_info': self.camera_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RGBDImage':
        """Create from dictionary"""
        return cls(
            rgb=np.array(data['rgb']),
            depth=np.array(data['depth']),
            timestamp=data['timestamp'],
            frame_id=data['frame_id'],
            camera_info=data['camera_info']
        )

@dataclass
class GraspPose:
    """6-DOF grasp pose representation"""
    position: Tuple[float, float, float]    # (x, y, z) in meters
    orientation: Tuple[float, float, float] # (rx, ry, rz) in radians
    confidence: float                       # Prediction confidence [0, 1]
    timestamp: float                        # Prediction timestamp
    
    def to_list(self) -> List[float]:
        """Convert to flat list format"""
        return list(self.position) + list(self.orientation)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'position': self.position,
            'orientation': self.orientation,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_list(cls, pose_list: List[float], confidence: float = 0.0) -> 'GraspPose':
        """Create from flat list [x, y, z, rx, ry, rz]"""
        return cls(
            position=(pose_list[0], pose_list[1], pose_list[2]),
            orientation=(pose_list[3], pose_list[4], pose_list[5]),
            confidence=confidence,
            timestamp=time.time()
        )

@dataclass
class RobotState:
    """Robot state information"""
    joint_names: List[str]
    joint_positions: List[float]    # Joint angles in radians
    joint_velocities: List[float]   # Joint velocities in rad/s
    joint_efforts: List[float]      # Joint torques in Nm
    end_effector_pose: Optional[GraspPose]
    gripper_state: bool            # True = closed, False = open
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'joint_names': self.joint_names,
            'joint_positions': self.joint_positions,
            'joint_velocities': self.joint_velocities,
            'joint_efforts': self.joint_efforts,
            'end_effector_pose': self.end_effector_pose.to_dict() if self.end_effector_pose else None,
            'gripper_state': self.gripper_state,
            'timestamp': self.timestamp
        }

@dataclass
class ExecutionResult:
    """Result of grasp execution"""
    success: bool
    executed_pose: GraspPose
    final_robot_state: RobotState
    execution_time: float          # Time to execute in seconds
    error_message: Optional[str]   # Error description if failed
    collision_detected: bool       # Whether collision occurred
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'executed_pose': self.executed_pose.to_dict(),
            'final_robot_state': self.final_robot_state.to_dict(),
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'collision_detected': self.collision_detected
        }

@dataclass
class TrainingEpisode:
    """Training episode data"""
    episode_id: int
    start_time: float
    end_time: Optional[float]
    success: bool
    total_reward: float
    action_count: int
    states: List[RGBDImage]
    actions: List[GraspPose]
    rewards: List[float]
    robot_states: List[RobotState]
    
    def duration(self) -> float:
        """Get episode duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'episode_id': self.episode_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'success': self.success,
            'total_reward': self.total_reward,
            'action_count': self.action_count,
            'duration': self.duration(),
            'states': [state.to_dict() for state in self.states],
            'actions': [action.to_dict() for action in self.actions],
            'rewards': self.rewards,
            'robot_states': [state.to_dict() for state in self.robot_states]
        }

class MessageType:
    """Message type constants"""
    CAMERA_DATA = "camera_data"
    GRASP_PREDICTION = "grasp_prediction"
    EXECUTION_FEEDBACK = "execution_feedback"
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    TRAINING_DATA = "training_data"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    DEBUG_INFO = "debug_info"

class NetworkMessage:
    """Base class for network messages"""
    
    def __init__(self, msg_type: str, data: Dict, timestamp: Optional[float] = None):
        self.type = msg_type
        self.data = data
        self.timestamp = timestamp or time.time()
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps({
            'type': self.type,
            'data': self.data,
            'timestamp': self.timestamp
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'NetworkMessage':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(
            msg_type=data['type'],
            data=data['data'],
            timestamp=data.get('timestamp')
        )

class CameraDataMessage(NetworkMessage):
    """Camera data message from VM to Host"""
    
    def __init__(self, rgbd_image: RGBDImage, robot_state: RobotState, episode_id: int):
        data = {
            'rgbd': rgbd_image.to_dict(),
            'robot_state': robot_state.to_dict(),
            'episode_id': episode_id
        }
        super().__init__(MessageType.CAMERA_DATA, data)

class GraspPredictionMessage(NetworkMessage):
    """Grasp prediction message from Host to VM"""
    
    def __init__(self, grasp_pose: GraspPose, processing_time: float):
        data = {
            'grasp_pose': grasp_pose.to_dict(),
            'processing_time': processing_time
        }
        super().__init__(MessageType.GRASP_PREDICTION, data)

class ExecutionFeedbackMessage(NetworkMessage):
    """Execution feedback message from VM to Host"""
    
    def __init__(self, result: ExecutionResult, episode_id: int):
        data = {
            'result': result.to_dict(),
            'episode_id': episode_id
        }
        super().__init__(MessageType.EXECUTION_FEEDBACK, data)

def compress_image_data(image: np.ndarray, quality: int = 80) -> str:
    """Compress image data for network transfer"""
    import cv2
    import base64
    
    if len(image.shape) == 3:  # RGB image
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:  # Depth image
        # Convert depth to 16-bit for better compression
        depth_16 = (image * 1000).astype(np.uint16)
        _, buffer = cv2.imencode('.png', depth_16)
    
    return base64.b64encode(buffer).decode('utf-8')

def decompress_image_data(compressed_data: str, is_depth: bool = False) -> np.ndarray:
    """Decompress image data from network transfer"""
    import cv2
    import base64
    
    buffer = base64.b64decode(compressed_data)
    nparr = np.frombuffer(buffer, np.uint8)
    
    if is_depth:
        # Decode depth image and convert back to float
        depth_16 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        return depth_16.astype(np.float32) / 1000.0
    else:
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

class DataValidator:
    """Validation utilities for data structures"""
    
    @staticmethod
    def validate_grasp_pose(pose: List[float]) -> bool:
        """Validate grasp pose format"""
        if len(pose) != 6:
            return False
        
        # Check position bounds (example: workspace limits)
        x, y, z = pose[0], pose[1], pose[2]
        if not (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0 and 0.0 <= z <= 1.0):
            return False
        
        # Check orientation bounds
        rx, ry, rz = pose[3], pose[4], pose[5]
        if not all(-np.pi <= angle <= np.pi for angle in [rx, ry, rz]):
            return False
        
        return True
    
    @staticmethod
    def validate_joint_positions(positions: List[float], joint_limits: List[Tuple[float, float]]) -> bool:
        """Validate robot joint positions"""
        if len(positions) != len(joint_limits):
            return False
        
        for pos, (min_limit, max_limit) in zip(positions, joint_limits):
            if not (min_limit <= pos <= max_limit):
                return False
        
        return True

# Configuration data structures
@dataclass
class NetworkConfig:
    """Network configuration"""
    host_ip: str = "192.168.1.1"
    host_port: int = 8888
    vm_ip: str = "192.168.1.100"
    timeout: int = 30
    retry_attempts: int = 5
    buffer_size: int = 4096
    heartbeat_interval: int = 10
    compression_enabled: bool = True
    compression_quality: int = 80

@dataclass  
class RobotConfig:
    """Robot configuration"""
    model: str = "UR3e"
    dof: int = 6
    joint_limits: List[Tuple[float, float]] = None
    max_joint_velocity: float = 3.14159
    max_joint_acceleration: float = 1.57
    workspace_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = None
    
    def __post_init__(self):
        if self.joint_limits is None:
            # Default UR3 joint limits
            self.joint_limits = [(-3.14159, 3.14159)] * 6
        
        if self.workspace_bounds is None:
            # Default workspace bounds (min_xyz, max_xyz)
            self.workspace_bounds = ((-0.8, -0.8, 0.0), (0.8, 0.8, 1.2))

@dataclass
class CameraConfig:
    """Camera configuration"""
    resolution: Tuple[int, int] = (640, 480)
    frame_rate: int = 30
    depth_range: Tuple[float, float] = (0.1, 2.0)
    rgb_topic: str = "/camera/image_raw"
    depth_topic: str = "/camera/depth/image_raw"

# Export commonly used types
__all__ = [
    'RGBDImage', 'GraspPose', 'RobotState', 'ExecutionResult', 'TrainingEpisode',
    'MessageType', 'NetworkMessage', 'CameraDataMessage', 'GraspPredictionMessage', 
    'ExecutionFeedbackMessage', 'compress_image_data', 'decompress_image_data',
    'DataValidator', 'NetworkConfig', 'RobotConfig', 'CameraConfig'
]

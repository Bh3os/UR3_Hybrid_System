#!/usr/bin/env python3
"""
Enhanced Webots Bridge for UR3 Hybrid System
Integrates with the actual Webots simulation environment
Based on the original supervisor.py and SimCamera.py implementations
"""

import os
import sys
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

try:
    from controller import Supervisor, Robot
    from scipy.spatial.transform import Rotation as Rot
    WEBOTS_AVAILABLE = True
except ImportError:
    WEBOTS_AVAILABLE = False
    print("Webots controller not available, using mock mode")

try:
    import rospy
    from std_msgs.msg import Int8
    from integrator.msg import BlockPose
    from integrator.srv import SupervisorGrabService, SupervisorPositionService
    from integrator.srv import SimImageCameraService, SimDepthCameraService
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    # Mock ROS types
    class MockROS:
        pass
    BlockPose = MockROS
    Image = MockROS

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

class WebotsSupervisor:
    """
    Enhanced Webots Supervisor for UR3 simulation control
    Based on the original WebotsSupervisor class with improvements
    """
    
    def __init__(self, simulation: bool = True, world_file: str = "Environment.wbt"):
        """
        Initialize Webots supervisor
        
        Args:
            simulation: Whether running in simulation mode (for testing)
            world_file: Name of the Webots world file to load
        """
        self.simulation = simulation
        self.logger = logging.getLogger('WebotsSupervisor')
        
        # Simulation parameters
        self.number_of_blocks = 5
        self.timestep = 4  # Default timestep in ms
        
        # UR3e robot parameters (from original)
        self.ur3e_position = [0.69, 0.74, 0]
        self.ur3e_rotation = None
        
        # Initialize Webots connection
        if WEBOTS_AVAILABLE and not simulation:
            self._init_webots_supervisor()
        else:
            self._init_mock_supervisor()
            
        # Initialize ROS services if available
        if ROS_AVAILABLE and not simulation:
            self._init_ros_services()
            
    def _init_webots_supervisor(self):
        """Initialize actual Webots supervisor connection"""
        try:
            # Set controller name
            os.environ['WEBOTS_ROBOT_NAME'] = 'supervisor'
            
            # Initialize supervisor
            self.supervisor = Supervisor()
            self.timestep = int(self.supervisor.getBasicTimeStep())
            
            # Set up robot rotation
            self.ur3e_rotation = Rot.from_rotvec(-(np.pi / 2) * np.array([1.0, 0.0, 0.0]))
            
            # Get all blocks in simulation
            self.blocks = []
            for i in range(self.number_of_blocks):
                block = self.supervisor.getFromDef(f"block{i}")
                if block:
                    self.blocks.append(block)
                else:
                    self.logger.warning(f"Block {i} not found in simulation")
                    
            # Get end-effector reference
            self.end_effector = self.supervisor.getFromDef("gps")
            
            self.logger.info(f"Webots supervisor initialized with {len(self.blocks)} blocks")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Webots supervisor: {e}")
            self._init_mock_supervisor()
            
    def _init_mock_supervisor(self):
        """Initialize mock supervisor for testing"""
        self.supervisor = None
        self.blocks = []
        self.end_effector = None
        
        # Create mock blocks
        for i in range(self.number_of_blocks):
            mock_block = {
                'id': i,
                'position': [np.random.uniform(-0.5, 0.5), 
                           np.random.uniform(-0.5, 0.5),
                           np.random.uniform(0.7, 0.9)],
                'rotation': [0, 0, np.random.uniform(0, 2*np.pi)]
            }
            self.blocks.append(mock_block)
            
        self.logger.info(f"Mock supervisor initialized with {len(self.blocks)} blocks")
        
    def _init_ros_services(self):
        """Initialize ROS services for Webots integration"""
        try:
            if not rospy.get_node_uri():
                rospy.init_node('webots_supervisor', anonymous=True)
                
            # Create ROS services
            self.grab_service = rospy.Service(
                'supervisor_grab_service', 
                SupervisorGrabService, 
                self._handle_grab_request
            )
            
            self.position_service = rospy.Service(
                'supervisor_position_service',
                SupervisorPositionService,
                self._handle_position_request  
            )
            
            self.logger.info("ROS services initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ROS services: {e}")
            
    def step(self) -> bool:
        """
        Step the simulation forward
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.supervisor:
            return self.supervisor.step(self.timestep) != -1
        else:
            # Mock simulation step
            time.sleep(self.timestep / 1000.0)  # Convert ms to seconds
            return True
            
    def get_block_poses(self) -> List[Dict[str, Any]]:
        """
        Get positions and orientations of all blocks
        
        Returns:
            List of block pose dictionaries
        """
        block_poses = []
        
        if self.supervisor and hasattr(self.supervisor, 'getFromDef'):
            # Real Webots implementation
            for i, block in enumerate(self.blocks):
                if block:
                    try:
                        position = block.getPosition()
                        rotation = block.getOrientation()
                        
                        block_poses.append({
                            'id': i,
                            'position': list(position) if position else [0, 0, 0],
                            'rotation': list(rotation) if rotation else [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            'timestamp': time.time()
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to get pose for block {i}: {e}")
        else:
            # Mock implementation
            for i, block in enumerate(self.blocks):
                if isinstance(block, dict):
                    block_poses.append({
                        'id': i,
                        'position': block['position'],
                        'rotation': block['rotation'] + [1, 0, 0, 0, 1, 0],  # Pad to 9 elements
                        'timestamp': time.time()
                    })
                    
        return block_poses
        
    def set_block_pose(self, block_id: int, position: List[float], 
                      rotation: Optional[List[float]] = None) -> bool:
        """
        Set the position and rotation of a specific block
        
        Args:  
            block_id: ID of the block to move
            position: [x, y, z] position
            rotation: [rx, ry, rz] rotation (optional)
            
        Returns:
            bool: True if successful
        """
        if block_id >= len(self.blocks):
            self.logger.error(f"Block ID {block_id} out of range")
            return False
            
        if self.supervisor and hasattr(self.supervisor, 'getFromDef'):
            # Real Webots implementation
            block = self.blocks[block_id]
            if block:
                try:
                    # Set position
                    block.getField('translation').setSFVec3f(position)
                    
                    # Set rotation if provided
                    if rotation:
                        # Convert rotation to Webots format if needed
                        block.getField('rotation').setSFRotation(rotation + [1.0])  # Add angle
                        
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to set block {block_id} pose: {e}")
                    return False
        else:
            # Mock implementation
            if isinstance(self.blocks[block_id], dict):
                self.blocks[block_id]['position'] = position
                if rotation:
                    self.blocks[block_id]['rotation'] = rotation
                return True
                
        return False
        
    def get_robot_state(self) -> Dict[str, Any]:
        """
        Get current robot state from simulation
        
        Returns:
            Dictionary containing robot state information
        """
        robot_state = {
            'position': self.ur3e_position.copy(),
            'rotation': [0, 0, 0],
            'joint_angles': [0.0] * 6,
            'end_effector_pose': [0, 0, 0, 0, 0, 0],
            'timestamp': time.time()
        }
        
        if self.supervisor and self.end_effector:
            try:
                # Get end-effector position
                ee_pos = self.end_effector.getPosition()
                if ee_pos:
                    robot_state['end_effector_pose'][:3] = list(ee_pos)
                    
                # Get end-effector orientation  
                ee_rot = self.end_effector.getOrientation()
                if ee_rot:
                    # Convert rotation matrix to euler angles
                    rot_matrix = np.array(ee_rot).reshape(3, 3)
                    if WEBOTS_AVAILABLE:
                        euler = Rot.from_matrix(rot_matrix).as_euler('xyz')
                        robot_state['end_effector_pose'][3:] = list(euler)
                        
            except Exception as e:
                self.logger.warning(f"Failed to get robot state: {e}")
                
        return robot_state
        
    def reset_simulation(self) -> bool:
        """
        Reset the simulation to initial state
        
        Returns:
            bool: True if successful
        """
        if self.supervisor:
            try:
                self.supervisor.simulationReset()
                return True
            except Exception as e:
                self.logger.error(f"Failed to reset simulation: {e}")
                return False
        else:
            # Mock reset - randomize block positions
            for block in self.blocks:
                if isinstance(block, dict):
                    block['position'] = [
                        np.random.uniform(-0.5, 0.5),
                        np.random.uniform(-0.5, 0.5), 
                        np.random.uniform(0.7, 0.9)
                    ]
                    block['rotation'] = [0, 0, np.random.uniform(0, 2*np.pi)]
            return True
            
    def _handle_grab_request(self, request):
        """Handle ROS grab service request"""
        # Implementation for grab service
        return True
        
    def _handle_position_request(self, request):
        """Handle ROS position service request"""  
        # Implementation for position service
        return self.get_robot_state()


class WebotsCamera:
    """
    Enhanced Webots Camera for image and depth acquisition
    Based on the original SimCamera class
    """
    
    def __init__(self, simulation: bool = True):
        """
        Initialize Webots camera
        
        Args:
            simulation: Whether running in simulation mode
        """
        self.simulation = simulation
        self.logger = logging.getLogger('WebotsCamera')
        
        # Camera parameters
        self.timestep = 4  # 4ms = 250 FPS
        self.image_width = 640
        self.image_height = 480
        
        # Initialize Webots camera
        if WEBOTS_AVAILABLE and not simulation:
            self._init_webots_camera()
        else:
            self._init_mock_camera()
            
        # Initialize ROS services if available
        if ROS_AVAILABLE and not simulation:
            self._init_ros_services()
            
    def _init_webots_camera(self):
        """Initialize actual Webots camera"""
        try:
            # Set camera robot name
            os.environ['WEBOTS_ROBOT_NAME'] = 'camera'
            
            # Initialize robot controller
            self.robot = Robot()
            self.timestep = int(self.robot.getBasicTimeStep())
            
            # Get camera devices by name for robustness
            self.camera = self.robot.getDevice('camera')  # Replace 'camera' with the actual RGB camera device name in Webots
            self.depth_camera = self.robot.getDevice('depth_camera')  # Replace 'depth_camera' with the actual depth camera device name in Webots
            
            # Enable cameras
            if self.camera:
                self.camera.enable(self.timestep)
                self.image_width = self.camera.getWidth()
                self.image_height = self.camera.getHeight()
                
            if self.depth_camera:
                self.depth_camera.enable(self.timestep)
                
            self.logger.info(f"Webots camera initialized ({self.image_width}x{self.image_height})")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Webots camera: {e}")
            self._init_mock_camera()
            
    def _init_mock_camera(self):
        """Initialize mock camera for testing"""
        self.robot = None
        self.camera = None
        self.depth_camera = None
        self.logger.info("Mock camera initialized")
        
    def _init_ros_services(self):
        """Initialize ROS services for camera"""
        try:
            if ROS_AVAILABLE:
                if not rospy.get_node_uri():
                    rospy.init_node('webots_camera', anonymous=True)
                    
                # Initialize CV bridge
                self.bridge = CvBridge()
                
                # Create ROS services
                self.image_service = rospy.Service(
                    'image_camera_service',
                    SimImageCameraService,
                    self._handle_image_request
                )
                
                self.depth_service = rospy.Service(
                    'depth_camera_service', 
                    SimDepthCameraService,
                    self._handle_depth_request
                )
                
                self.logger.info("Camera ROS services initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize camera ROS services: {e}")
            
    def capture_rgb_image(self) -> Optional[np.ndarray]:
        """
        Capture RGB image from camera
        
        Returns:
            RGB image as numpy array or None if failed
        """
        if self.camera and WEBOTS_AVAILABLE:
            try:
                # Step simulation to get fresh image
                if self.robot:
                    self.robot.step(self.timestep)
                    
                # Get image data
                image_data = self.camera.getImageArray()
                if image_data:
                    # Convert to numpy array and proper format
                    image = np.array(image_data, dtype=np.uint8)
                    # Webots images are in BGR format, convert to RGB
                    if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if OPENCV_AVAILABLE else image
                    return image
                    
            except Exception as e:
                self.logger.error(f"Failed to capture RGB image: {e}")
                
        # Mock image generation
        image = np.random.randint(0, 255, 
                                (self.image_height, self.image_width, 3))
        return image.astype(np.uint8)
        
    def capture_depth_image(self) -> Optional[np.ndarray]:
        """
        Capture depth image from camera
        
        Returns:
            Depth image as numpy array or None if failed
        """
        if self.depth_camera and WEBOTS_AVAILABLE:
            try:
                # Step simulation to get fresh image
                if self.robot:
                    self.robot.step(self.timestep)
                    
                # Get depth data
                depth_data = self.depth_camera.getRangeImageArray()
                if depth_data:
                    # Convert to numpy array
                    depth = np.array(depth_data, dtype=np.float32)
                    return depth
                    
            except Exception as e:
                self.logger.error(f"Failed to capture depth image: {e}")
                
        # Mock depth generation
        depth = np.random.uniform(0.1, 2.0, 
                                (self.image_height, self.image_width))
        return depth.astype(np.float32)
        
    def capture_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture both RGB and depth images
        
        Returns:
            Tuple of (rgb_image, depth_image)
        """
        rgb_image = self.capture_rgb_image()
        depth_image = self.capture_depth_image()
        return rgb_image, depth_image
        
    def _handle_image_request(self, request):
        """Handle ROS image service request"""
        rgb_image = self.capture_rgb_image()
        if rgb_image is not None and ROS_AVAILABLE:
            try:
                ros_image = self.bridge.cv2_to_imgmsg(rgb_image, "rgb8")
                return ros_image
            except Exception as e:
                self.logger.error(f"Failed to convert image to ROS message: {e}")
        return None
        
    def _handle_depth_request(self, request):
        """Handle ROS depth service request"""
        depth_image = self.capture_depth_image()
        if depth_image is not None and ROS_AVAILABLE:
            try:
                ros_depth = self.bridge.cv2_to_imgmsg(depth_image, "32FC1")
                return ros_depth
            except Exception as e:
                self.logger.error(f"Failed to convert depth to ROS message: {e}")
        return None


class WebotsBridge:
    """
    Main Webots Bridge combining supervisor and camera functionality
    This is the main interface for the UR3 Hybrid System
    """
    
    def __init__(self, simulation: bool = True, world_file: str = "Environment.wbt"):
        """
        Initialize complete Webots bridge
        
        Args:
            simulation: Whether running in simulation mode
            world_file: Webots world file to use
        """
        self.simulation = simulation
        self.logger = logging.getLogger('WebotsBridge')
        
        # Initialize components
        self.supervisor = WebotsSupervisor(simulation, world_file)
        self.camera = WebotsCamera(simulation)
        
        self.logger.info(f"Webots bridge initialized (simulation={simulation})")
        
    def step(self) -> bool:
        """Step the simulation forward"""
        return self.supervisor.step()
        
    def get_block_poses(self) -> List[Dict[str, Any]]:
        """Get all block poses"""
        return self.supervisor.get_block_poses()
        
    def get_robot_state(self) -> Dict[str, Any]:
        """Get robot state"""
        return self.supervisor.get_robot_state()
        
    def capture_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture RGB and depth images"""
        return self.camera.capture_rgbd()
        
    def reset_simulation(self) -> bool:
        """Reset simulation"""
        return self.supervisor.reset_simulation()
        
    def set_block_pose(self, block_id: int, position: List[float], 
                      rotation: Optional[List[float]] = None) -> bool:
        """Set block pose"""
        return self.supervisor.set_block_pose(block_id, position, rotation)


# Factory function for integration with the existing system
def create_webots_bridge(config: Optional[Dict[str, Any]] = None,
                        simulation: bool = True) -> WebotsBridge:
    """
    Factory function to create Webots bridge
    
    Args:
        config: Configuration dictionary (optional)
        simulation: Whether to run in simulation mode
        
    Returns:
        WebotsBridge instance
    """
    world_file = "Environment.wbt"
    if config and 'world_file' in config:
        world_file = config['world_file']
        
    return WebotsBridge(simulation=simulation, world_file=world_file)

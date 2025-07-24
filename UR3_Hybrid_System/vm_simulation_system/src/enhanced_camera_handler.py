#!/usr/bin/env python3
"""
Enhanced Camera Handler for UR3 System
Integrates RGB-D camera processing with ROS and Webots simulation
Based on the original camera.py implementation
"""

import numpy as np
import cv2
import time
import yaml
import os
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
from collections import deque
import logging

try:
    import rospy
    from cv_bridge import CvBridge, CvBridgeError
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import Point, PointStamped
    from std_msgs.msg import Header
    import pyrealsense2 as rs2
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS or RealSense not available, using simulation mode")
    # Create mock classes for type hints when ROS is not available
    class Image:
        pass
    class CameraInfo:
        pass
    class Header:
        pass
    class CvBridge:
        pass

# Local imports for Webots integration
try:
    from integrator.srv import SimImageCameraService, SimDepthCameraService
    WEBOTS_AVAILABLE = True
except ImportError:
    WEBOTS_AVAILABLE = False
    print("Webots integration not available")
    # Create mock service classes
    class SimImageCameraService:
        pass
    class SimDepthCameraService:
        pass

class EnhancedCameraHandler:
    """
    Enhanced camera handler supporting multiple camera types and processing modes
    """
    
    def __init__(self, config_path: str = "config/camera_config.yaml", 
                 simulation: bool = True, camera_type: str = "realsense"):
        """
        Initialize camera handler
        
        Args:
            config_path: Path to camera configuration file
            simulation: Whether running in simulation mode
            camera_type: Type of camera ('realsense', 'kinect', 'simulation')
        """
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Camera parameters
        self.is_sim = simulation
        self.camera_type = camera_type
        self.image_size = tuple(self.config.get('image_size', [480, 640]))
        self.h, self.w = self.image_size
        
        # Image processing parameters
        self.depth_scale = self.config.get('depth_scale', 0.001)  # mm to m
        self.depth_max = self.config.get('depth_max', 2.0)  # Maximum depth in meters
        self.depth_min = self.config.get('depth_min', 0.1)   # Minimum depth in meters
        
        # Camera intrinsics (will be loaded or estimated)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.depth_camera_matrix = None
        
        # Current frames
        self.current_rgb_frame = None
        self.current_depth_frame = None
        self.current_aligned_depth = None
        self.frame_timestamp = None
        
        # Processing history for stability
        self.distances = deque(maxlen=10)
        self.rgb_history = deque(maxlen=3)
        self.depth_history = deque(maxlen=3)
        
        # ROS integration
        if ROS_AVAILABLE:
            self.bridge = CvBridge()
            self._setup_ros_interface()
        
        # Hardware camera initialization
        self.camera_pipeline = None
        if not simulation:
            self._initialize_hardware_camera()
        
        # Setup logging
        self.logger = logging.getLogger('CameraHandler')
        self.logger.info(f"Camera handler initialized - Type: {camera_type}, Simulation: {simulation}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load camera configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Handle empty or None config
                if not config or not isinstance(config, dict):
                    config = {}
                # Validate and set defaults for required keys
                default_config = self._get_default_config()
                for key, value in default_config.items():
                    if key not in config or not isinstance(config.get(key), type(value)):
                        config[key] = value
                return config
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default camera configuration"""
        return {
            'image_size': [480, 640],
            'fps': 30,
            'depth_scale': 0.001,
            'depth_max': 2.0,
            'depth_min': 0.1,
            'camera_intrinsics': {
                'fx': 525.0, 'fy': 525.0,
                'cx': 320.0, 'cy': 240.0
            },
            'processing': {
                'bilateral_filter': True,
                'temporal_filter': True,
                'spatial_filter': True,
                'hole_filling': True
            }
        }
    
    def _setup_ros_interface(self):
        """Setup ROS publishers and subscribers"""
        # Publishers
        self.rgb_pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=1)
        self.aligned_depth_pub = rospy.Publisher('/camera/aligned_depth/image_raw', Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher('/camera/rgb/camera_info', CameraInfo, queue_size=1)
        
        # Subscribers for external camera feeds
        self.external_rgb_sub = rospy.Subscriber(
            '/external_camera/rgb', Image, self._external_rgb_callback
        )
        self.external_depth_sub = rospy.Subscriber(
            '/external_camera/depth', Image, self._external_depth_callback
        )
        
        # Service clients for Webots simulation
        if WEBOTS_AVAILABLE:
            try:
                rospy.wait_for_service('/sim_image_camera_service', timeout=5.0)
                rospy.wait_for_service('/sim_depth_camera_service', timeout=5.0)
                
                self.sim_rgb_service = rospy.ServiceProxy(
                    '/sim_image_camera_service', SimImageCameraService
                )
                self.sim_depth_service = rospy.ServiceProxy(
                    '/sim_depth_camera_service', SimDepthCameraService
                )
                
                self.logger.info("Connected to Webots camera services")
            except rospy.ROSException:
                self.logger.warning("Could not connect to Webots camera services")
    
    def _initialize_hardware_camera(self):
        """Initialize hardware camera (RealSense)"""
        if self.camera_type == "realsense":
            try:
                self.camera_pipeline = rs2.pipeline()
                config = rs2.config()
                
                # Configure streams
                config.enable_stream(rs2.stream.color, self.w, self.h, rs2.format.bgr8, 30)
                config.enable_stream(rs2.stream.depth, self.w, self.h, rs2.format.z16, 30)
                
                # Start pipeline
                profile = self.camera_pipeline.start(config)
                
                # Get camera intrinsics
                color_profile = profile.get_stream(rs2.stream.color)
                color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
                
                self.camera_matrix = np.array([
                    [color_intrinsics.fx, 0, color_intrinsics.ppx],
                    [0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0, 0, 1]
                ])
                
                self.dist_coeffs = np.array(color_intrinsics.coeffs)
                
                # Setup alignment
                self.align = rs2.align(rs2.stream.color)
                
                # Setup filters
                self._setup_depth_filters()
                
                self.logger.info("RealSense camera initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize RealSense camera: {e}")
                self.camera_pipeline = None
    
    def _setup_depth_filters(self):
        """Setup RealSense depth filters for improved quality"""
        if not hasattr(self, 'camera_pipeline') or self.camera_pipeline is None:
            return
            
        # Spatial filter
        self.spatial_filter = rs2.spatial_filter()
        self.spatial_filter.set_option(rs2.option.filter_magnitude, 2)
        self.spatial_filter.set_option(rs2.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs2.option.filter_smooth_delta, 20)
        
        # Temporal filter
        self.temporal_filter = rs2.temporal_filter()
        self.temporal_filter.set_option(rs2.option.filter_smooth_alpha, 0.4)
        self.temporal_filter.set_option(rs2.option.filter_smooth_delta, 20)
        
        # Hole filling filter
        self.hole_filling_filter = rs2.hole_filling_filter()
        
        self.logger.info("Depth filters configured")
    
    def _external_rgb_callback(self, msg: Image):
        """Handle external RGB camera feed"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_rgb_frame = cv_image
            self.frame_timestamp = msg.header.stamp
            self.rgb_history.append(cv_image.copy())
        except CvBridgeError as e:
            self.logger.error(f"Error converting RGB image: {e}")
    
    def _external_depth_callback(self, msg: Image):
        """Handle external depth camera feed"""
        try:
            # Handle different depth encodings
            if msg.encoding == "16UC1":
                cv_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                # Convert to meters
                depth_image = cv_image.astype(np.float32) * self.depth_scale
            elif msg.encoding == "32FC1":
                depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                self.logger.warning(f"Unsupported depth encoding: {msg.encoding}")
                return
            
            self.current_depth_frame = depth_image
            self.depth_history.append(depth_image.copy())
            
        except CvBridgeError as e:
            self.logger.error(f"Error converting depth image: {e}")
    
    def capture_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture RGB and depth frames from active camera source
        
        Returns:
            Tuple of (rgb_frame, depth_frame) or (None, None) if failed
        """
        
        if self.is_sim:
            return self._capture_simulation_frames()
        elif self.camera_pipeline is not None:
            return self._capture_hardware_frames()
        else:
            # Use ROS feeds if available
            return self.current_rgb_frame, self.current_depth_frame
    
    def _capture_simulation_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture frames from Webots simulation"""
        if not WEBOTS_AVAILABLE or not hasattr(self, 'sim_rgb_service'):
            # Generate synthetic frames for testing
            rgb_frame = np.random.randint(0, 255, (self.h, self.w, 3), dtype=np.uint8)
            depth_frame = np.random.uniform(0.5, 2.0, (self.h, self.w)).astype(np.float32)
            return rgb_frame, depth_frame
        
        try:
            # Get RGB frame from Webots
            rgb_response = self.sim_rgb_service()
            if rgb_response.success:
                rgb_array = np.array(rgb_response.image_data, dtype=np.uint8)
                rgb_frame = rgb_array.reshape((self.h, self.w, 3))
                rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            else:
                rgb_frame = None
            
            # Get depth frame from Webots
            depth_response = self.sim_depth_service()
            if depth_response.success:
                depth_array = np.array(depth_response.depth_data, dtype=np.float32)
                depth_frame = depth_array.reshape((self.h, self.w))
            else:
                depth_frame = None
            
            return rgb_frame, depth_frame
            
        except rospy.ServiceException as e:
            self.logger.error(f"Service call failed: {e}")
            return None, None
    
    def _capture_hardware_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture frames from hardware camera (RealSense)"""
        try:
            # Wait for frames
            frames = self.camera_pipeline.wait_for_frames(timeout_ms=1000)
            
            # Align depth to color
            aligned_frames = self.align.process(frames)
            
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            # Apply filters to depth
            if hasattr(self, 'spatial_filter'):
                depth_frame = self.spatial_filter.process(depth_frame)
                depth_frame = self.temporal_filter.process(depth_frame)
                depth_frame = self.hole_filling_filter.process(depth_frame)
            
            # Convert to numpy arrays
            rgb_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Convert depth to meters
            depth_image = depth_image.astype(np.float32) * self.depth_scale
            
            # Apply depth limits
            depth_image = np.where(
                (depth_image < self.depth_min) | (depth_image > self.depth_max),
                0, depth_image
            )
            
            return rgb_image, depth_image
            
        except Exception as e:
            self.logger.error(f"Error capturing hardware frames: {e}")
            return None, None
    
    def process_frames(self, rgb_frame: np.ndarray, depth_frame: np.ndarray) -> Dict[str, Any]:
        """
        Process captured frames for object detection and analysis
        
        Args:
            rgb_frame: RGB image array
            depth_frame: Depth image array
            
        Returns:
            Dictionary containing processed data
        """
        processed_data = {
            'rgb_frame': rgb_frame,
            'depth_frame': depth_frame,
            'timestamp': time.time()
        }
        
        if rgb_frame is None or depth_frame is None:
            return processed_data
        
        # Basic image processing
        processed_data.update({
            'rgb_enhanced': self._enhance_rgb_image(rgb_frame),
            'depth_filtered': self._filter_depth_image(depth_frame),
            'depth_colormap': self._create_depth_colormap(depth_frame),
            'segmentation_mask': self._simple_segmentation(rgb_frame, depth_frame)
        })
        
        # Object detection and analysis
        objects = self._detect_objects(rgb_frame, depth_frame)
        processed_data['detected_objects'] = objects
        
        # Calculate grasp candidates
        if objects:
            grasp_candidates = self._calculate_grasp_candidates(objects, depth_frame)
            processed_data['grasp_candidates'] = grasp_candidates
        
        return processed_data
    
    def _enhance_rgb_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """Apply enhancement to RGB image"""
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply bilateral filter for noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def _filter_depth_image(self, depth_image: np.ndarray) -> np.ndarray:
        """Apply filtering to depth image"""
        # Handle invalid depth values
        filtered_depth = depth_image.copy()
        filtered_depth[filtered_depth == 0] = np.nan
        
        # Apply median filter to reduce noise
        filtered_depth = cv2.medianBlur(filtered_depth.astype(np.float32), 5)
        
        # Fill small holes using inpainting
        mask = np.isnan(filtered_depth).astype(np.uint8)
        if np.any(mask):
            # Replace NaNs with zeros for inpainting
            depth_for_inpaint = filtered_depth.copy()
            depth_for_inpaint[np.isnan(depth_for_inpaint)] = 0
            filtered_depth = cv2.inpaint(
                depth_for_inpaint, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
            )
            # Optionally, restore NaNs where mask was set
            filtered_depth[mask == 1] = np.nan
        
        return filtered_depth
    
    def _create_depth_colormap(self, depth_image: np.ndarray) -> np.ndarray:
        """Create colormap visualization of depth image"""
        # Normalize depth for visualization
        depth_normalized = (depth_image - self.depth_min) / (self.depth_max - self.depth_min)
        depth_normalized = np.clip(depth_normalized, 0, 1)
        
        # Convert to 8-bit
        depth_8bit = (depth_normalized * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        
        return depth_colormap
    
    def _simple_segmentation(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        """Perform simple object segmentation"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        # Create depth mask (objects in certain depth range)
        depth_mask = ((depth_image > 0.3) & (depth_image < 1.5)).astype(np.uint8) * 255
        
        # Create color-based mask (this is simplified - in practice you'd use learned features)
        # Here we're looking for objects that are not background color
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Combine masks
        combined_mask = cv2.bitwise_and(depth_mask, binary_mask)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def _detect_objects(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> List[Dict]:
        """Detect objects in the scene"""
        objects = []
        
        # Get segmentation mask
        mask = self._simple_segmentation(rgb_image, depth_image)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < 500:  # Minimum object size
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate object center
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Get depth at object center
            if (0 <= center_y < depth_image.shape[0] and 
                0 <= center_x < depth_image.shape[1]):
                object_depth = depth_image[center_y, center_x]
            else:
                continue
            
            # Convert pixel coordinates to 3D position
            if self.camera_matrix is not None:
                world_pos = self._pixel_to_world(center_x, center_y, object_depth)
            else:
                world_pos = [0, 0, object_depth]
            
            # Create object dictionary
            obj = {
                'id': i,
                'bbox': [x, y, w, h],
                'center_2d': [center_x, center_y],
                'center_3d': world_pos,
                'area': area,
                'depth': object_depth,
                'contour': contour
            }
            
            objects.append(obj)
        
        return objects
    
    def _calculate_grasp_candidates(self, objects: List[Dict], depth_image: np.ndarray) -> List[Dict]:
        """Calculate grasp candidates for detected objects"""
        grasp_candidates = []
        
        for obj in objects:
            # Get object contour
            contour = obj['contour']
            
            # Calculate object orientation
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
            else:
                angle = 0
            
            # Calculate multiple grasp poses around the object
            center_3d = obj['center_3d']
            
            # Generate grasp candidates at different angles
            for grasp_angle in [0, 45, 90, 135]:
                grasp_orientation = angle + grasp_angle
                
                # Calculate approach vector (simplified)
                approach_vector = [0, 0, -1]  # Top-down grasp
                
                grasp_candidate = {
                    'object_id': obj['id'],
                    'position': center_3d,
                    'orientation': grasp_orientation,
                    'approach_vector': approach_vector,
                    'quality_score': 0.5,  # Placeholder - would be calculated by ML model
                    'grasp_type': 'top_down'
                }
                
                grasp_candidates.append(grasp_candidate)
        
        # Sort by quality score
        grasp_candidates.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return grasp_candidates
    
    def _pixel_to_world(self, u: int, v: int, depth: float) -> List[float]:
        """Convert pixel coordinates to world coordinates"""
        if self.camera_matrix is None:
            return [0, 0, depth]
        
        # Get camera intrinsics
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        
        # Convert to world coordinates
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        return [x, y, z]
    
    def publish_frames(self, rgb_frame: np.ndarray, depth_frame: np.ndarray):
        """Publish frames to ROS topics"""
        if not ROS_AVAILABLE:
            return
        
        try:
            # Create header
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera_link"
            
            # Publish RGB frame
            if rgb_frame is not None:
                rgb_msg = self.bridge.cv2_to_imgmsg(rgb_frame, "bgr8")
                rgb_msg.header = header
                self.rgb_pub.publish(rgb_msg)
            
            # Publish depth frame
            if depth_frame is not None:
                # Convert to millimeters for publishing
                depth_mm = (depth_frame * 1000).astype(np.uint16)
                depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, "16UC1")
                depth_msg.header = header
                self.depth_pub.publish(depth_msg)
            
            # Publish camera info
            self._publish_camera_info(header)
            
        except CvBridgeError as e:
            self.logger.error(f"Error publishing frames: {e}")
    
    def _publish_camera_info(self, header: Header):
        """Publish camera calibration info"""
        if self.camera_matrix is None:
            return
        
        camera_info = CameraInfo()
        camera_info.header = header
        camera_info.width = self.w
        camera_info.height = self.h
        
        # Set camera matrix
        camera_info.K = self.camera_matrix.flatten().tolist()
        
        # Set distortion coefficients
        if self.dist_coeffs is not None:
            camera_info.D = self.dist_coeffs.tolist()
        
        # Set rectification and projection matrices (identity for simple case)
        camera_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        camera_info.P = self.camera_matrix.flatten().tolist() + [0, 0, 0, 0]
        
        self.camera_info_pub.publish(camera_info)
    
    def get_average_distance(self, center_x: int, center_y: int, radius: int = 10) -> float:
        """Get average distance in a region around a point"""
        if self.current_depth_frame is None:
            return 0.0
        
        # Define region of interest
        y1, y2 = max(0, center_y - radius), min(self.h, center_y + radius)
        x1, x2 = max(0, center_x - radius), min(self.w, center_x + radius)
        
        # Extract region
        region = self.current_depth_frame[y1:y2, x1:x2]
        
        # Calculate average excluding zeros
        valid_depths = region[region > 0]
        if len(valid_depths) > 0:
            return np.mean(valid_depths)
        else:
            return 0.0
    
    def cleanup(self):
        """Cleanup camera resources"""
        if hasattr(self, 'camera_pipeline') and self.camera_pipeline is not None:
            self.camera_pipeline.stop()
            self.logger.info("Camera pipeline stopped")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()


def create_camera_system(config_path: str = "config/camera_config.yaml",
                        simulation: bool = True,
                        camera_type: str = "realsense") -> EnhancedCameraHandler:
    """
    Factory function to create camera system
    
    Args:
        config_path: Configuration file path
        simulation: Whether to run in simulation mode
        camera_type: Type of camera to use
        
    Returns:
        EnhancedCameraHandler instance
    """
    return EnhancedCameraHandler(config_path, simulation, camera_type)

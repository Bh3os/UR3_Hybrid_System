#!/usr/bin/env python3
"""
VM Simulation Client for UR3 System
Runs on Ubuntu VM and communicates with Windows host GPU server
Handles Webots simulation, ROS integration, and robot control
"""

import socket
import json
import numpy as np
import cv2
import time
import threading
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ROS imports
try:
    import rospy
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float32MultiArray, Bool, Empty
    from geometry_msgs.msg import Pose, PoseStamped
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from cv_bridge import CvBridge, CvBridgeError
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS not available, using simulation mode")
    # Mock ROS classes for type hints
    class CvBridge:
        pass
    class Image:
        pass
    class JointState:
        pass
    class Float32MultiArray:
        pass
    class Bool:
        pass
    class Empty:
        pass
    class Pose:
        pass
    class PoseStamped:
        pass

# Local imports
from enhanced_robot_controller import UR3KinematicsController, GripperController, MotionPlanner, create_robot_system
from enhanced_camera_handler import EnhancedCameraHandler, create_camera_system
from webots_bridge import WebotsBridge

class SimulationClient:
    """Main client for VM simulation system"""
    
    def __init__(self, config_path: str = "config/network_config.yaml"):
        """Initialize the simulation client"""
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize ROS node if available
        if ROS_AVAILABLE:
            rospy.init_node('ur3_simulation_client', anonymous=True)
            rospy.loginfo("UR3 Simulation Client started")
        else:
            print("UR3 Simulation Client started (ROS not available)")
        
        # Network connection
        self.host_socket = None
        self.connected = False
        self.connection_lock = threading.Lock()
        
        # Initialize components
        self.bridge = CvBridge() if ROS_AVAILABLE else None
        
        # Create robot system components
        self.robot_controller, self.gripper_controller, self.motion_planner = create_robot_system(
            config_path="config/robot_config.yaml",
            simulation=True
        )
        
        # Create camera handler
        self.camera_handler = EnhancedCameraHandler(
            config_path="config/camera_config.yaml",
            simulation=True,
            camera_type="simulation"
        )
        
        self.webots_bridge = WebotsBridge(simulation=True)
        
        # Data storage
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.latest_joint_states = None
        self.episode_data = []
        
        # Setup ROS publishers and subscribers
        self._setup_ros_interface()
        
        # Episode management
        self.episode_count = 0
        self.episode_start_time = None
        self.episode_active = False
        
        rospy.loginfo("Simulation client initialization complete")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            rospy.logwarn(f"Config file {config_path} not found, using defaults")
            return {
                'network': {
                    'host_ip': '192.168.1.1',
                    'host_port': 8888,
                    'timeout': 30,
                    'retry_attempts': 5
                }
            }
    
    def _setup_ros_interface(self):
        """Setup ROS publishers and subscribers"""
        
        # Subscribers
        self.rgb_sub = rospy.Subscriber(
            '/camera/image_raw', Image, self._rgb_callback, queue_size=1
        )
        self.depth_sub = rospy.Subscriber(
            '/camera/depth/image_raw', Image, self._depth_callback, queue_size=1
        )
        self.joint_state_sub = rospy.Subscriber(
            '/ur3/joint_states', JointState, self._joint_state_callback
        )
        
        # Publishers
        self.joint_cmd_pub = rospy.Publisher(
            '/ur3/joint_commands', JointTrajectory, queue_size=1
        )
        self.gripper_cmd_pub = rospy.Publisher(
            '/gripper/command', Bool, queue_size=1
        )
        self.episode_reset_pub = rospy.Publisher(
            '/simulation/reset', Empty, queue_size=1
        )
        
        # Service clients for robot control
        rospy.wait_for_service('/ur3/get_pose', timeout=10)
        
        rospy.loginfo("ROS interface setup complete")
    
    def _rgb_callback(self, msg: Image):
        """Handle RGB camera data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_rgb_image = cv_image
            self._try_process_camera_data()
        except CvBridgeError as e:
            rospy.logerr(f"RGB callback error: {e}")
    
    def _depth_callback(self, msg: Image):
        """Handle depth camera data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            self.latest_depth_image = cv_image
            self._try_process_camera_data()
        except CvBridgeError as e:
            rospy.logerr(f"Depth callback error: {e}")
    
    def _joint_state_callback(self, msg: JointState):
        """Handle robot joint state updates"""
        self.latest_joint_states = {
            'names': msg.name,
            'positions': list(msg.position),
            'velocities': list(msg.velocity),
            'efforts': list(msg.effort)
        }
    
    def _try_process_camera_data(self):
        """Process camera data when both RGB and depth are available"""
        if self.latest_rgb_image is not None and self.latest_depth_image is not None:
            if self.connected:
                self._send_camera_data_to_host()
    
    def connect_to_host(self) -> bool:
        """Establish connection to host GPU server"""
        host_ip = self.config['network']['host_ip']
        host_port = self.config['network']['host_port']
        timeout = self.config['network']['timeout']
        
        try:
            with self.connection_lock:
                self.host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.host_socket.settimeout(timeout)
                self.host_socket.connect((host_ip, host_port))
                self.connected = True
                
            rospy.loginfo(f"Connected to host GPU server at {host_ip}:{host_port}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Failed to connect to host: {e}")
            self.connected = False
            if self.host_socket:
                self.host_socket.close()
            return False
    
    def _send_camera_data_to_host(self):
        """Send RGBD camera data to host for processing"""
        try:
            import base64

            # Encode RGB image as JPEG
            _, rgb_encoded = cv2.imencode('.jpg', self.latest_rgb_image)
            rgb_b64 = base64.b64encode(rgb_encoded.tobytes()).decode('utf-8')

            # Encode depth image as PNG (preserves float32 if needed)
            depth_to_send = self.latest_depth_image
            if depth_to_send.dtype != np.uint16:
                # Convert to 16-bit for PNG if needed
                depth_to_send = (depth_to_send * 1000).astype(np.uint16)
            _, depth_encoded = cv2.imencode('.png', depth_to_send)
            depth_b64 = base64.b64encode(depth_encoded.tobytes()).decode('utf-8')

            # Prepare camera data
            camera_data = {
                'type': 'camera_data',
                'data': {
                    'rgb': rgb_b64,
                    'depth': depth_b64,
                    'timestamp': time.time(),
                    'episode': self.episode_count
                },
                'robot_state': self.latest_joint_states
            }
            
            # Send to host
            response = self._send_message_to_host(camera_data)
            
            if response and response.get('type') == 'grasp_prediction':
                self._execute_grasp_prediction(response)
                
        except Exception as e:
            rospy.logerr(f"Error sending camera data: {e}")
            self.connected = False
    
    def _send_message_to_host(self, message: Dict) -> Optional[Dict]:
        """Send message to host and receive response"""
        if not self.connected:
            return None
        
        try:
            with self.connection_lock:
                # Send message
                message_data = json.dumps(message).encode('utf-8')
                self.host_socket.send(len(message_data).to_bytes(4, byteorder='big'))
                self.host_socket.send(message_data)
                
                # Receive response
                response_size = int.from_bytes(self.host_socket.recv(4), byteorder='big')
                response_data = b''
                while len(response_data) < response_size:
                    chunk = self.host_socket.recv(min(response_size - len(response_data), 4096))
                    if not chunk:
                        break
                    response_data += chunk
                
                response = json.loads(response_data.decode('utf-8'))
                return response
                
        except Exception as e:
            rospy.logerr(f"Communication error: {e}")
            self.connected = False
            return None
    
    def _execute_grasp_prediction(self, prediction: Dict):
        """Execute grasp prediction from host"""
        try:
            pose = prediction['pose']
            confidence = prediction.get('confidence', 0.0)
            
            rospy.loginfo(f"Executing grasp: pose={pose[:3]}, confidence={confidence:.3f}")
            
            # Convert pose to robot commands
            success = self.robot_controller.execute_grasp(pose)
            
            # Send feedback to host
            feedback = {
                'type': 'execution_feedback',
                'success': success,
                'pose': pose,
                'timestamp': time.time(),
                'episode': self.episode_count
            }
            
            self._send_message_to_host(feedback)
            
            # Store episode data
            self.episode_data.append({
                'prediction': prediction,
                'execution': feedback,
                'joint_states': self.latest_joint_states
            })
            
        except Exception as e:
            rospy.logerr(f"Error executing grasp: {e}")
    
    def start_new_episode(self):
        """Start a new training episode"""
        self.episode_count += 1
        self.episode_start_time = time.time()
        self.episode_active = True
        self.episode_data = []
        
        # Reset simulation environment
        self.webots_bridge.reset_simulation()
        self.robot_controller.home_position()
        
        rospy.loginfo(f"Started episode {self.episode_count}")
        
        # Notify host about new episode
        episode_info = {
            'type': 'episode_start',
            'episode': self.episode_count,
            'timestamp': self.episode_start_time
        }
        self._send_message_to_host(episode_info)
    
    def end_current_episode(self, success: bool = False):
        """End the current training episode"""
        if not self.episode_active:
            return
        
        episode_duration = time.time() - self.episode_start_time
        self.episode_active = False
        
        # Calculate episode statistics
        episode_summary = {
            'type': 'episode_end',
            'episode': self.episode_count,
            'duration': episode_duration,
            'success': success,
            'actions_count': len(self.episode_data),
            'timestamp': time.time()
        }
        
        # Send to host
        self._send_message_to_host(episode_summary)
        
        rospy.loginfo(f"Episode {self.episode_count} ended - Duration: {episode_duration:.2f}s, Success: {success}")
        
        # Save episode data locally
        self._save_episode_data()
    
    def _save_episode_data(self):
        """Save episode data to local storage"""
        try:
            data_dir = Path("data/episodes")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            episode_file = data_dir / f"episode_{self.episode_count}.json"
            
            episode_info = {
                'episode': self.episode_count,
                'start_time': self.episode_start_time,
                'end_time': time.time(),
                'data': self.episode_data
            }
            
            with open(episode_file, 'w') as f:
                json.dump(episode_info, f, indent=2)
                
            rospy.logdebug(f"Saved episode data: {episode_file}")
            
        except Exception as e:
            rospy.logerr(f"Error saving episode data: {e}")
    
    def run_simulation_loop(self):
        """Main simulation loop"""
        rospy.loginfo("Starting simulation loop...")
        
        # Connect to host
        retry_attempts = self.config['network']['retry_attempts']
        connected = False
        
        for attempt in range(retry_attempts):
            if self.connect_to_host():
                connected = True
                break
            else:
                rospy.logwarn(f"Connection attempt {attempt + 1}/{retry_attempts} failed")
                time.sleep(2)
        
        if not connected:
            rospy.logerr("Failed to connect to host after all attempts")
            return
        
        # Start first episode
        self.start_new_episode()
        
        # Main loop
        rate = rospy.Rate(10)  # 10 Hz
        
        try:
            while not rospy.is_shutdown():
                # Check connection status
                if not self.connected:
                    rospy.logwarn("Connection lost, attempting to reconnect...")
                    if not self.connect_to_host():
                        rospy.logwarn("Reconnection failed, retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                
                # Episode management (example: end episode after 60 seconds)
                if self.episode_active and (time.time() - self.episode_start_time) > 60:
                    self.end_current_episode(success=False)  # Timeout
                    time.sleep(2)  # Brief pause between episodes
                    self.start_new_episode()
                
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.loginfo("Simulation interrupted by user")
        except Exception as e:
            rospy.logerr(f"Simulation loop error: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources"""
        rospy.loginfo("Cleaning up simulation client...")
        
        # End current episode if active
        if self.episode_active:
            self.end_current_episode(success=False)
        
        # Close host connection
        if self.host_socket:
            try:
                self.host_socket.close()
            except:
                pass
        
        rospy.loginfo("Simulation client cleanup complete")

def main():
    """Main entry point"""
    try:
        # Create and run simulation client
        client = SimulationClient()
        rospy.loginfo("🤖 VM Simulation Client initialized")
        
        # Start simulation loop
        client.run_simulation_loop()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS shutdown requested")
    except Exception as e:
        rospy.logerr(f"Client error: {e}")
        raise

if __name__ == "__main__":
    main()

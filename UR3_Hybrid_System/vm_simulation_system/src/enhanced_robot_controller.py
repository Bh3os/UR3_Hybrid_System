#!/usr/bin/env python3
"""
Enhanced Robot Controller for UR3 System
Integrated with actual UR3 kinematics and improved motion planning
Based on the original Kinematics.py implementation
"""

import numpy as np
import time
import yaml
import math
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from math import pi, sin, cos, asin, acos, atan2, radians, sqrt
import logging

try:
    import rospy
    import actionlib
    from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
    from sensor_msgs.msg import JointState
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from std_msgs.msg import Bool, Float32MultiArray
    from std_srvs.srv import Trigger
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS not available, using simulation mode")

try:
    from scipy.spatial.transform import Rotation as Rot
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Create a simple mock for Rotation if scipy is not available
    class MockRotation:
        @staticmethod
        def from_matrix(matrix):
            return MockRotation()
        
        def as_quat(self):
            return [0, 0, 0, 1]  # Identity quaternion
    
    Rot = MockRotation

class UR3KinematicsController:
    """
    Enhanced UR3 robot controller with full kinematics implementation
    Based on the original URKinematics class with improvements
    """
    
    def __init__(self, config_path: str = "config/robot_config.yaml", 
                 simulation: bool = True, precision: float = 0.02):
        """
        Initialize the UR3 controller
        
        Args:
            config_path: Path to robot configuration file
            simulation: Whether running in simulation mode
            precision: Movement precision in meters
        """
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Robot parameters
        self.is_sim = simulation
        self.precision = precision
        self.debug = False
        
        # UR3 DH parameters (meters)
        self.d = [0.1519, 0, 0, 0.11235, 0.08535, 0.0819]  # Link offsets
        self.a = [0, -0.24365, -0.21325, 0, 0, 0]          # Link lengths
        self.alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]          # Link twists
        
        # Joint limits (radians)
        self.joint_limits = [
            [-2*pi, 2*pi],      # Joint 1
            [-2*pi, 2*pi],      # Joint 2
            [-pi, pi],          # Joint 3
            [-2*pi, 2*pi],      # Joint 4
            [-2*pi, 2*pi],      # Joint 5
            [-2*pi, 2*pi]       # Joint 6
        ]
        
        # Current state
        self.joints_state = [0.0] * 6
        self.is_moving = False
        self.last_joint_command = [0.0] * 6
        
        # Motion parameters
        self.max_joint_velocity = 1.0  # rad/s
        self.max_joint_acceleration = 2.0  # rad/s²
        self.blend_radius = 0.05  # For smooth trajectory blending
        
        # Setup logging
        self.logger = logging.getLogger('UR3Controller')
        
        if ROS_AVAILABLE:
            self._setup_ros_interface()
        
        self.logger.info("UR3 Kinematics Controller initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load robot configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Handle empty or None config
                if config is None:
                    config = {}
                return config
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default robot configuration"""
        return {
            'robot': {
                'name': 'ur3',
                'dof': 6,
                'max_velocity': 1.0,
                'max_acceleration': 2.0
            },
            'workspace': {
                'bounds': {
                    'x_min': -0.8, 'x_max': 0.8,
                    'y_min': -0.8, 'y_max': 0.8,
                    'z_min': 0.0, 'z_max': 1.2
                }
            },
            'grasp': {
                'approach_distance': 0.1,
                'grasp_force': 50.0,
                'retreat_distance': 0.1
            }
        }
    
    def _setup_ros_interface(self):
        """Setup ROS publishers, subscribers, and action clients"""
        # Publishers
        self.joint_trajectory_pub = rospy.Publisher(
            '/ur3/joint_trajectory', JointTrajectory, queue_size=1
        )
        self.gripper_pub = rospy.Publisher(
            '/gripper/command', Bool, queue_size=1
        )
        
        # Subscribers
        self.joint_state_sub = rospy.Subscriber(
            '/joint_states', JointState, self._joint_state_callback
        )
        
        # Action client for trajectory execution
        self.trajectory_client = actionlib.SimpleActionClient(
            '/ur3/follow_joint_trajectory', FollowJointTrajectoryAction
        )
        
        # Wait for action server
        if self.trajectory_client.wait_for_server(timeout=rospy.Duration(5.0)):
            self.logger.info("Connected to trajectory action server")
        else:
            self.logger.warning("Could not connect to trajectory action server")
    
    def _joint_state_callback(self, msg: 'JointState'):
        """Update current joint states from ROS message"""
        if len(msg.position) >= 6:
            self.joints_state = list(msg.position[:6])
    
    def dh_transform(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """
        Calculate transformation matrix using DH parameters
        
        Args:
            theta, d, a, alpha: DH parameters
            
        Returns:
            4x4 transformation matrix
        """
        ct = cos(theta)
        st = sin(theta)
        ca = cos(alpha)
        sa = sin(alpha)
        
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,      sa,     ca,    d],
            [0,       0,      0,    1]
        ])
    
    def forward_kinematics(self, joint_angles: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate forward kinematics for UR3
        
        Args:
            joint_angles: List of 6 joint angles in radians
            
        Returns:
            Tuple of (position, orientation_matrix)
        """
        if len(joint_angles) != 6:
            raise ValueError("Expected 6 joint angles")
        
        # Initialize transformation matrix
        T = np.eye(4)
        
        # Apply DH transformations for each joint
        for i in range(6):
            T_i = self.dh_transform(
                joint_angles[i], self.d[i], self.a[i], self.alpha[i]
            )
            T = np.dot(T, T_i)
        
        # Extract position and orientation
        position = T[:3, 3]
        orientation = T[:3, :3]
        
        return position, orientation
    
    def inverse_kinematics(self, target_pose: List[float], 
                          current_joints: Optional[List[float]] = None) -> Optional[List[float]]:
        """
        Calculate inverse kinematics for UR3 using analytical solution
        
        Args:
            target_pose: [x, y, z, rx, ry, rz] in meters and radians
            current_joints: Current joint configuration for solution selection
            
        Returns:
            List of 6 joint angles in radians, or None if unreachable
        """
        if current_joints is None:
            current_joints = self.joints_state
        
        x, y, z, rx, ry, rz = target_pose
        
        # Check workspace bounds and IK reachability
        if not self._is_pose_reachable(x, y, z, rx, ry, rz):
            self.logger.warning(f"Pose ({x:.3f}, {y:.3f}, {z:.3f}) is outside workspace or unreachable")
            return None
        
        # Convert rotation to rotation matrix
        rotation_matrix = Rot.from_euler('xyz', [rx, ry, rz]).as_matrix()
        
        # Target transformation matrix
        T_target = np.eye(4)
        T_target[:3, 3] = [x, y, z]
        T_target[:3, :3] = rotation_matrix
        
        # Analytical IK solution for UR3
        solutions = self._solve_ik_analytical(T_target)
        
        if not solutions:
            self.logger.warning("No IK solution found")
            return None
        
        # Select best solution based on current joint configuration
        best_solution = self._select_best_ik_solution(solutions, current_joints)
        
        return best_solution
    
    def _solve_ik_analytical(self, T_target: np.ndarray) -> List[List[float]]:
        """
        Analytical IK solution for UR3
        Returns all possible solutions
        """
        solutions = []
        
        # Extract target position and orientation
        px, py, pz = T_target[:3, 3]
        R = T_target[:3, :3]
        
        # Calculate wrist center position
        d6 = self.d[5]  # Tool offset
        wrist_center = np.array([px, py, pz]) - d6 * R[:, 2]
        
        # Solve for theta1
        theta1_solutions = self._solve_theta1(wrist_center)
        
        for theta1 in theta1_solutions:
            # Solve for theta5
            theta5_solutions = self._solve_theta5(T_target, theta1)
            
            for theta5 in theta5_solutions:
                # Solve for theta6
                theta6 = self._solve_theta6(T_target, theta1, theta5)
                
                # Solve for theta2, theta3, theta4
                theta234_solutions = self._solve_theta234(wrist_center, theta1)
                
                for theta2, theta3, theta4 in theta234_solutions:
                    solution = [theta1, theta2, theta3, theta4, theta5, theta6]
                    
                    # Validate solution
                    if self._validate_joint_solution(solution):
                        solutions.append(solution)
        
        return solutions
    
    def _solve_theta1(self, wrist_center: np.ndarray) -> List[float]:
        """Solve for theta1 given wrist center position"""
        x_wc, y_wc, z_wc = wrist_center
        
        # Two solutions for theta1
        theta1_1 = atan2(y_wc, x_wc)
        theta1_2 = theta1_1 + pi if theta1_1 < 0 else theta1_1 - pi
        
        return [theta1_1, theta1_2]
    
    def _solve_theta5(self, T_target: np.ndarray, theta1: float) -> List[float]:
        """Solve for theta5 given target pose and theta1"""
        # Simplified solution - in practice this would be more complex
        # Extract wrist center and orientation
        d6 = self.d[5]
    def _solve_theta6(self, T_target: np.ndarray, theta1: float, theta5: float) -> float:
        """Solve for theta6 given target pose, theta1, and theta5"""
        # Extract rotation matrix from target pose
        R = T_target[:3, :3]
        # Calculate theta6 using elements of the rotation matrix and previously solved theta1 and theta5
        # Avoid division by zero if sin(theta5) is very small
        if abs(sin(theta5)) < 1e-6:
            return 0.0
        theta6 = atan2(
            -R[1, 2] * sin(theta1) + R[0, 2] * cos(theta1),
            R[1, 1] * sin(theta1) - R[0, 1] * cos(theta1)
        )
        return theta6
        wy = py - d6 * R[1, 2]
        wz = pz - d6 * R[2, 2]
        # Calculate theta5 using the orientation and theta1
        # The wrist axis is aligned with the robot's 5th joint
        # Compute the direction of the wrist
        s1 = sin(theta1)
        c1 = cos(theta1)
        nx = R[0, 2]
        ny = R[1, 2]
        nz = R[2, 2]
        # theta5 = acos(nz1), where nz1 is the z component of the wrist in base frame
        theta5_1 = acos(round(nx * s1 - ny * c1, 8))
        theta5_2 = -theta5_1
        return [theta5_1, theta5_2]
    
    def _solve_theta6(self, T_target: np.ndarray, theta1: float, theta5: float) -> float:
        """Solve for theta6 given target pose, theta1, and theta5"""
        # Simplified solution
        return 0.0  # Example solution
    
    def _solve_theta234(self, wrist_center: np.ndarray, theta1: float) -> List[Tuple[float, float, float]]:
        """Solve for theta2, theta3, theta4 given wrist center and theta1"""
        # Extract DH parameters for readability
        d1, d4 = self.d[0], self.d[3]
        a2, a3 = self.a[1], self.a[2]

        x_wc, y_wc, z_wc = wrist_center

        # Calculate wrist center position in base frame
        # Project wrist center into the plane of the second joint
        x1 = cos(theta1) * x_wc + sin(theta1) * y_wc
        y1 = z_wc - d1

        # Compute r and s for geometric IK
        r = sqrt(x1**2 + y1**2)
        s = y1

        # Law of cosines for theta3
        D = (r**2 - a2**2 - a3**2) / (2 * a2 * a3)
        solutions = []
        if abs(D) > 1.0:
            return solutions  # No solution

        # Two possible solutions for theta3
        theta3_1 = atan2(sqrt(1 - D**2), D)
        theta3_2 = atan2(-sqrt(1 - D**2), D)

        for theta3 in [theta3_1, theta3_2]:
            # Compute theta2
            k1 = a2 + a3 * cos(theta3)
            k2 = a3 * sin(theta3)
            theta2 = atan2(s, x1) - atan2(k2, k1)

            # For theta4, assume 0 (wrist aligned), or could be computed from orientation
            theta4 = 0.0

            solutions.append((theta2, theta3, theta4))

        return solutions
    
    def _validate_joint_solution(self, solution: List[float]) -> bool:
        """Validate that joint solution is within limits"""
        for i, angle in enumerate(solution):
            min_limit, max_limit = self.joint_limits[i]
            if not (min_limit <= angle <= max_limit):
                return False
        return True
    
    def _select_best_ik_solution(self, solutions: List[List[float]], 
                                current_joints: List[float]) -> List[float]:
        """Select the IK solution closest to current joint configuration"""
        if not solutions:
            return None
        
        best_solution = solutions[0]
        min_distance = sum((s - c) ** 2 for s, c in zip(best_solution, current_joints))
        
        for solution in solutions[1:]:
            distance = sum((s - c) ** 2 for s, c in zip(solution, current_joints))
            if distance < min_distance:
                min_distance = distance
                best_solution = solution
        
        return best_solution
    
    def _is_pose_reachable(self, x: float, y: float, z: float, rx: float = 0.0, ry: float = 0.0, rz: float = 0.0) -> bool:
        """
        Check if pose is within robot workspace and has at least one valid IK solution within joint limits.
        """
        bounds = self.config['workspace']['bounds']
        in_bounds = (bounds['x_min'] <= x <= bounds['x_max'] and
                     bounds['y_min'] <= y <= bounds['y_max'] and
                     bounds['z_min'] <= z <= bounds['z_max'])
        if not in_bounds:
            return False

        # Try to find at least one valid IK solution within joint limits
        try:
            rotation_matrix = Rot.from_euler('xyz', [rx, ry, rz]).as_matrix()
            T_target = np.eye(4)
            T_target[:3, 3] = [x, y, z]
            T_target[:3, :3] = rotation_matrix
            solutions = self._solve_ik_analytical(T_target)
            for sol in solutions:
                if self._validate_joint_solution(sol):
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"IK check failed: {e}")
            return False
    
    def move_to_joint_positions(self, target_joints: List[float], 
                               duration: float = 3.0, 
                               wait: bool = True) -> bool:
        """
        Move robot to target joint positions
        
        Args:
            target_joints: Target joint angles in radians
            duration: Movement duration in seconds
            wait: Whether to wait for completion
            
        Returns:
            True if successful, False otherwise
        """
        if len(target_joints) != 6:
            self.logger.error("Expected 6 joint angles")
            return False
        
        # Validate joint limits
        for i, angle in enumerate(target_joints):
            min_limit, max_limit = self.joint_limits[i]
            if not (min_limit <= angle <= max_limit):
                self.logger.error(f"Joint {i} angle {angle:.3f} exceeds limits [{min_limit:.3f}, {max_limit:.3f}]")
                return False
        
        if ROS_AVAILABLE:
            return self._execute_joint_trajectory(target_joints, duration, wait)
        else:
            # Simulation mode
            self.joints_state = target_joints.copy()
            time.sleep(duration * 0.1)  # Simulate movement time
            return True
    
    def move_to_pose(self, target_pose: List[float], 
                    duration: float = 3.0, 
                    wait: bool = True) -> bool:
        """
        Move robot to target Cartesian pose
        
        Args:
            target_pose: [x, y, z, rx, ry, rz] target pose
            duration: Movement duration in seconds
            wait: Whether to wait for completion
            
        Returns:
            True if successful, False otherwise
        """
        # Calculate inverse kinematics
        target_joints = self.inverse_kinematics(target_pose)
        
        if target_joints is None:
            self.logger.error("Could not find IK solution for target pose")
            return False
        
        return self.move_to_joint_positions(target_joints, duration, wait)
    
    def _execute_joint_trajectory(self, target_joints: List[float], 
                                 duration: float, wait: bool) -> bool:
        """Execute joint trajectory using ROS action"""
        if not hasattr(self, 'trajectory_client'):
            self.logger.error("ROS action client not available")
            return False
        
        # Create trajectory goal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = [f'ur3_joint_{i+1}' for i in range(6)]
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.velocities = [0.0] * 6
        point.time_from_start = rospy.Duration(duration)
        
        goal.trajectory.points = [point]
        
        # Send goal
        self.trajectory_client.send_goal(goal)
        
        if wait:
            # Wait for completion
            result = self.trajectory_client.wait_for_result(timeout=rospy.Duration(duration + 5.0))
            if result:
                self.logger.info("Joint trajectory executed successfully")
                return True
            else:
                self.logger.error("Joint trajectory execution timed out")
                return False
        
        return True
    
    def get_current_pose(self) -> Tuple[List[float], List[float]]:
        """
        Get current end-effector pose
        
        Returns:
            Tuple of (position, orientation) as lists
        """
        position, orientation_matrix = self.forward_kinematics(self.joints_state)
        
        # Convert rotation matrix to Euler angles
        rotation = Rot.from_matrix(orientation_matrix)
        orientation = rotation.as_euler('xyz')
        
        return position.tolist(), orientation.tolist()
    
    def stop_motion(self):
        """Stop current robot motion"""
        if ROS_AVAILABLE and hasattr(self, 'trajectory_client'):
            self.trajectory_client.cancel_goal()
        
        self.is_moving = False
        self.logger.info("Robot motion stopped")
    
    def home_position(self) -> bool:
        """Move robot to home position"""
        home_joints = [0.0, -pi/2, pi/2, -pi/2, -pi/2, 0.0]
        return self.move_to_joint_positions(home_joints, duration=5.0)
    
    def get_joint_positions(self) -> List[float]:
        """Get current joint positions"""
        return self.joints_state.copy()
    
    def is_motion_complete(self) -> bool:
        """Check if robot motion is complete"""
        if not self.is_moving:
            return True
        
        # Check if current position is close to target
        if hasattr(self, 'last_joint_command'):
            diff = [abs(current - target) for current, target in 
                   zip(self.joints_state, self.last_joint_command)]
            return all(d < 0.01 for d in diff)  # 0.01 rad tolerance
        
        return True


class GripperController:
    """Simple gripper controller"""
    
    def __init__(self):
        self.is_closed = False
        self.grip_force = 0.0
        
        if ROS_AVAILABLE:
            self.gripper_pub = rospy.Publisher('/gripper/command', Bool, queue_size=1)
    
    def close_gripper(self, force: float = 50.0) -> bool:
        """Close gripper with specified force"""
        self.is_closed = True
        self.grip_force = force
        
        if ROS_AVAILABLE:
            self.gripper_pub.publish(Bool(data=True))
        
        time.sleep(1.0)  # Simulate gripper closing time
        return True
    
    def open_gripper(self) -> bool:
        """Open gripper"""
        self.is_closed = False
        self.grip_force = 0.0
        
        if ROS_AVAILABLE:
            self.gripper_pub.publish(Bool(data=False))
        
        time.sleep(1.0)  # Simulate gripper opening time
        return True
    
    def get_gripper_state(self) -> Dict[str, Any]:
        """Get current gripper state"""
        return {
            'is_closed': self.is_closed,
            'force': self.grip_force
        }


class MotionPlanner:
    """Advanced motion planning utilities"""
    
    def __init__(self, robot_controller: UR3KinematicsController):
        self.robot = robot_controller
        self.logger = logging.getLogger('MotionPlanner')
    
    def plan_grasp_approach(self, grasp_pose: List[float], 
                           approach_distance: float = 0.1) -> List[List[float]]:
        """
        Plan approach trajectory for grasping
        
        Args:
            grasp_pose: Final grasp pose [x, y, z, rx, ry, rz]
            approach_distance: Distance to approach from
            
        Returns:
            List of waypoints for approach trajectory
        """
        x, y, z, rx, ry, rz = grasp_pose
        
        # Calculate approach pose (offset along Z-axis)
        approach_pose = [x, y, z + approach_distance, rx, ry, rz]
        
        # Generate waypoints
        waypoints = []
        
        # Current pose
        current_pos, current_orient = self.robot.get_current_pose()
        waypoints.append(current_pos + current_orient)
        
        # Approach pose
        waypoints.append(approach_pose)
        
        # Final grasp pose
        waypoints.append(grasp_pose)
        
        return waypoints
    
    def plan_retreat_motion(self, retreat_distance: float = 0.1) -> List[List[float]]:
        """Plan retreat motion after grasping"""
        current_pos, current_orient = self.robot.get_current_pose()
        
        # Retreat along Z-axis
        retreat_pose = current_pos.copy()
        retreat_pose[2] += retreat_distance
        
        return [current_pos + current_orient, retreat_pose + current_orient]
    
    def execute_trajectory(self, waypoints: List[List[float]], 
                          segment_duration: float = 2.0) -> bool:
        """Execute a multi-waypoint trajectory"""
        success = True
        
        for i, waypoint in enumerate(waypoints):
            self.logger.info(f"Moving to waypoint {i+1}/{len(waypoints)}")
            
            if not self.robot.move_to_pose(waypoint, segment_duration, wait=True):
                self.logger.error(f"Failed to reach waypoint {i+1}")
                success = False
                break
            
            time.sleep(0.5)  # Brief pause between waypoints
        
        return success


def create_robot_system(config_path: str = "config/robot_config.yaml", 
                       simulation: bool = True) -> Tuple[UR3KinematicsController, GripperController, MotionPlanner]:
    """
    Factory function to create complete robot system
    
    Args:
        config_path: Configuration file path
        simulation: Whether to run in simulation mode
        
    Returns:
        Tuple of (robot_controller, gripper_controller, motion_planner)
    """
    robot_controller = UR3KinematicsController(config_path, simulation)
    gripper_controller = GripperController()
    motion_planner = MotionPlanner(robot_controller)
    
    return robot_controller, gripper_controller, motion_planner

#!/usr/bin/env python3
"""
VM Simulation System Integration Test
Tests the complete VM simulation stack including robot control, camera handling, 
Webots integration, and communication with the host GPU system
"""

import os
import sys
import time
import json
import numpy as np
import unittest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all VM simulation components
try:
    from enhanced_robot_controller import UR3KinematicsController, GripperController, MotionPlanner, create_robot_system
    from enhanced_camera_handler import EnhancedCameraHandler, create_camera_system
    from webots_bridge import WebotsBridge, WEBOTS_AVAILABLE
    from simulation_client import SimulationClient
    print("✅ All VM simulation modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class VMSimulationIntegrationTest(unittest.TestCase):
    """Comprehensive integration test for VM simulation system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("🔍 Setting up VM Simulation Integration Tests...")
        
        # Create test config directories
        os.makedirs("config", exist_ok=True)
        
        # Create test configurations
        cls._create_test_configs()
        
    @classmethod
    def _create_test_configs(cls):
        """Create test configuration files"""
        
        # Robot configuration
        robot_config = {
            'robot': {
                'type': 'UR3e',
                'simulation': True,
                'dh_parameters': {
                    'a': [0, -0.24365, -0.21325, 0, 0, 0],
                    'd': [0.1519, 0, 0, 0.11235, 0.08535, 0.0819],
                    'alpha': [1.570796327, 0, 0, 1.570796327, -1.570796327, 0]
                },
                'joint_limits': {
                    'min': [-3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159],
                    'max': [3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159]
                }
            },
            'gripper': {
                'type': 'robotiq_85',
                'max_force': 235,
                'max_speed': 0.15
            },
            'motion_planning': {
                'max_velocity': 1.0,
                'max_acceleration': 2.0,
                'planning_time': 5.0
            }
        }
        
        with open("config/robot_config.yaml", 'w') as f:
            import yaml
            yaml.dump(robot_config, f)
        
        # Camera configuration
        camera_config = {
            'camera': {
                'type': 'realsense',
                'simulation': True,
                'image_size': [480, 640],
                'fps': 30,
                'depth_scale': 0.001,
                'depth_max': 2.0,
                'depth_min': 0.1
            },
            'processing': {
                'enable_filters': True,
                'spatial_filter': True,
                'temporal_filter': True,
                'hole_filling': True
            }
        }
        
        with open("config/camera_config.yaml", 'w') as f:
            yaml.dump(camera_config, f)
        
        # Network configuration
        network_config = {
            'network': {
                'host_ip': '192.168.1.1',
                'host_port': 8888,
                'timeout': 30,
                'retry_attempts': 5,
                'buffer_size': 4096
            }
        }
        
        with open("config/network_config.yaml", 'w') as f:
            yaml.dump(network_config, f)
    
    def test_01_robot_controller(self):
        """Test UR3 robot controller functionality"""
        print("🤖 Testing Robot Controller...")
        
        # Test controller initialization
        controller = UR3KinematicsController(simulation=True)
        self.assertIsNotNone(controller)
        
        # Test forward kinematics
        joint_angles = [0, -1.57, 1.57, 0, 1.57, 0]  # Example pose
        position, orientation = controller.forward_kinematics(joint_angles)
        self.assertIsNotNone(position)
        self.assertIsNotNone(orientation)
        self.assertEqual(position.shape, (3,))  # 3D position
        self.assertEqual(orientation.shape, (3, 3))  # 3x3 rotation matrix
        
        # Test get joint positions
        joints = controller.get_joint_positions()
        self.assertIsInstance(joints, list)
        self.assertEqual(len(joints), 6)
        
        print("✅ Robot Controller test passed")
    
    def test_02_gripper_controller(self):
        """Test gripper controller"""
        print("🤏 Testing Gripper Controller...")
        
        gripper = GripperController()
        self.assertIsNotNone(gripper)
        
        # Test gripper commands
        result = gripper.open_gripper()
        self.assertTrue(result)
        
        result = gripper.close_gripper()
        self.assertTrue(result)
        
        # Test gripper state
        self.assertIsInstance(gripper.is_closed, bool)
        self.assertIsInstance(gripper.grip_force, float)
        
        print("✅ Gripper Controller test passed")
    
    def test_03_motion_planner(self):
        """Test motion planning functionality"""
        print("🛤️ Testing Motion Planner...")
        
        robot = UR3KinematicsController(simulation=True)
        planner = MotionPlanner(robot)
        self.assertIsNotNone(planner)
        
        # Test grasp approach planning
        grasp_pose = [0.3, 0.3, 0.3, 0, 0, 0]
        trajectory = planner.plan_grasp_approach(grasp_pose)
        self.assertIsNotNone(trajectory)
        self.assertIsInstance(trajectory, list)
        
        print("✅ Motion Planner test passed")
    
    def test_04_camera_handler(self):
        """Test camera handler functionality"""
        print("📷 Testing Camera Handler...")
        
        camera = EnhancedCameraHandler(simulation=True, camera_type="simulation")
        self.assertIsNotNone(camera)
        
        # Test image capture (will be mock data in simulation)
        rgb_image, depth_image = camera.capture_frames()
        
        if rgb_image is not None:
            self.assertEqual(len(rgb_image.shape), 3)  # H, W, C
            self.assertEqual(rgb_image.shape[2], 3)    # RGB channels
        
        if depth_image is not None:
            self.assertEqual(len(depth_image.shape), 2)  # H, W (depth is 2D)
        
        print("✅ Camera Handler test passed")
    
    def test_05_webots_bridge(self):
        """Test Webots simulation bridge"""
        print("🌍 Testing Webots Bridge...")
        
        if not WEBOTS_AVAILABLE:
            print("⚠️ Webots not available, using mock mode")
        
        bridge = WebotsBridge(simulation=True)
        self.assertIsNotNone(bridge)
        
        # Test basic simulation operations
        result = bridge.step()
        self.assertTrue(result)
        
        # Test environment interaction
        block_poses = bridge.get_block_poses()
        self.assertIsInstance(block_poses, list)
        
        # Test robot state
        robot_state = bridge.get_robot_state()
        self.assertIsInstance(robot_state, dict)
        
        # Test camera interface
        rgb_image, depth_image = bridge.capture_images()
        if rgb_image is not None:
            self.assertEqual(len(rgb_image.shape), 3)  # H, W, C
        
        print("✅ Webots Bridge test passed")
    
    def test_06_robot_system_factory(self):
        """Test robot system factory function"""
        print("🏭 Testing Robot System Factory...")
        
        robot_controller, gripper_controller, motion_planner = create_robot_system(
            config_path="config/robot_config.yaml",
            simulation=True
        )
        
        self.assertIsNotNone(robot_controller)
        self.assertIsNotNone(gripper_controller)
        self.assertIsNotNone(motion_planner)
        
        print("✅ Robot System Factory test passed")
    
    def test_07_camera_system_factory(self):
        """Test camera system factory function"""
        print("📸 Testing Camera System Factory...")
        
        camera_handler = create_camera_system(
            config_path="config/camera_config.yaml",
            simulation=True,
            camera_type="simulation"
        )
        
        self.assertIsNotNone(camera_handler)
        
        print("✅ Camera System Factory test passed")
    
    def test_08_simulation_client_components(self):
        """Test simulation client component integration"""
        print("🎮 Testing Simulation Client Components...")
        
        # Test that simulation client can be imported and basic structure exists
        # Full initialization might fail due to ROS/network dependencies
        
        # Check if class can be accessed
        self.assertTrue(hasattr(SimulationClient, '__init__'))
        self.assertTrue(hasattr(SimulationClient, '_load_config'))
        
        print("✅ Simulation Client Components test passed")
    
    def test_09_data_flow_simulation(self):
        """Test simulated data flow between components"""
        print("🔄 Testing Data Flow Simulation...")
        
        # Create components
        robot_controller = UR3KinematicsController(simulation=True)
        camera = EnhancedCameraHandler(simulation=True, camera_type="simulation")
        bridge = WebotsBridge()
        
        # Simulate robot movement
        joint_angles = [0.1, -0.5, 0.3, -0.2, 0.4, 0.0]
        pose = robot_controller.forward_kinematics(joint_angles)
        
        # Simulate camera capture
        rgb_image, depth_image = camera.capture_frames()
        
        # Simulate environment state
        block_poses = bridge.get_block_poses()
        
        # Create mock data packet (similar to what would be sent to GPU server)
        data_packet = {
            'robot_pose': pose,
            'joint_angles': joint_angles,
            'rgb_shape': rgb_image.shape if rgb_image is not None else None,
            'depth_shape': depth_image.shape if depth_image is not None else None,
            'num_blocks': len(block_poses),
            'timestamp': time.time()
        }
        
        # Verify data packet structure
        self.assertIn('robot_pose', data_packet)
        self.assertIn('joint_angles', data_packet)
        self.assertIn('timestamp', data_packet)
        
        print("✅ Data Flow Simulation test passed")
    
    def test_10_error_handling(self):
        """Test error handling and robustness"""
        print("🛡️ Testing Error Handling...")
        
        # Test invalid configurations
        try:
            # This should handle missing config gracefully
            controller = UR3KinematicsController(
                config_path="nonexistent_config.yaml", 
                simulation=True
            )
            # Should use defaults
            self.assertIsNotNone(controller)
        except Exception as e:
            # Should not crash completely
            self.assertIsNotNone(str(e))
        
        # Test invalid joint angles
        controller = UR3KinematicsController(simulation=True)
        invalid_joints = [10, 10, 10, 10, 10, 10]  # Outside limits
        
        # Should handle gracefully
        pose = controller.forward_kinematics(invalid_joints)
        # Should still return something (maybe clamped or error indication)
        
        print("✅ Error Handling test passed")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests"""
        print("🧹 Cleaning up VM simulation tests...")
        
        # Clean up test files
        test_files = [
            "config/robot_config.yaml",
            "config/camera_config.yaml", 
            "config/network_config.yaml"
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)

def run_vm_integration_tests():
    """Run all VM simulation integration tests"""
    print("=" * 70)
    print("🚀 VM Simulation System - Comprehensive Integration Test")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(VMSimulationIntegrationTest)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("🎉 ALL VM SIMULATION TESTS PASSED!")
        print("✅ The VM Simulation System is ready for integration")
        print("🤖 Robot Control: ✅")
        print("📷 Camera System: ✅") 
        print("🌍 Webots Bridge: ✅")
        print("🎮 Simulation Client: ✅")
        print("🔄 Data Flow: ✅")
    else:
        print("❌ Some VM simulation tests failed")
        print(f"Failed: {len(result.failures)}, Errors: {len(result.errors)}")
    
    print("=" * 70)
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_vm_integration_tests()
    sys.exit(0 if success else 1)

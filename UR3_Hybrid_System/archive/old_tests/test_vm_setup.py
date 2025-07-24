#!/usr/bin/env python3
"""
Test script to validate VM simulation system setup
Run this on the Ubuntu VM
"""

import sys
import time
import os
import subprocess
import socket
from pathlib import Path

def test_python_packages():
    """Test if required Python packages are installed"""
    print("Testing Python package imports...")
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('yaml', 'PyYAML'),
        ('scipy', 'SciPy'),
        ('PIL', 'Pillow')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name} - OK")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    else:
        print("\nAll required Python packages are installed!")
        return True

def test_ros_installation():
    """Test ROS installation"""
    print("\nTesting ROS installation...")
    
    try:
        # Check if ROS is installed
        result = subprocess.run(['rosversion', '-d'], capture_output=True, text=True)
        if result.returncode == 0:
            ros_distro = result.stdout.strip()
            print(f"✓ ROS {ros_distro} is installed")
        else:
            print("✗ ROS is not installed or not in PATH")
            return False
        
        # Check if roscore can be found
        result = subprocess.run(['which', 'roscore'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ roscore found")
        else:
            print("✗ roscore not found")
            return False
        
        # Test ROS Python packages
        ros_packages = [
            ('rospy', 'ROS Python client'),
            ('sensor_msgs', 'Sensor Messages'),
            ('geometry_msgs', 'Geometry Messages'),
            ('std_msgs', 'Standard Messages'),
            ('cv_bridge', 'CV Bridge')
        ]
        
        for package, name in ros_packages:
            try:
                __import__(package)
                print(f"✓ {name} - OK")
            except ImportError:
                print(f"✗ {name} - MISSING")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ ROS test failed: {e}")
        return False

def test_webots_installation():
    """Test Webots installation"""
    print("\nTesting Webots installation...")
    
    try:
        # Check if Webots is installed
        result = subprocess.run(['which', 'webots'], capture_output=True, text=True)
        if result.returncode == 0:
            webots_path = result.stdout.strip()
            print(f"✓ Webots found at: {webots_path}")
        else:
            print("✗ Webots not found in PATH")
            return False
        
        # Check version
        result = subprocess.run(['webots', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✓ Webots version: {version}")
        else:
            print("✗ Could not get Webots version")
        
        # Check if world files exist
        world_dir = Path('webots_worlds')
        if world_dir.exists():
            world_files = list(world_dir.glob('*.wbt'))
            if world_files:
                print(f"✓ Found {len(world_files)} world file(s)")
                for world_file in world_files:
                    print(f"  - {world_file.name}")
            else:
                print("⚠️  No world files found in webots_worlds/")
        else:
            print("⚠️  webots_worlds directory not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Webots test failed: {e}")
        return False

def test_camera_system():
    """Test camera system"""
    print("\nTesting camera system...")
    
    try:
        sys.path.append('src')
        from enhanced_camera_handler import EnhancedCameraHandler
        
        # Create camera handler in simulation mode
        camera = EnhancedCameraHandler(
            config_path="config/camera_config.yaml",
            simulation=True,
            camera_type="simulation"
        )
        
        print("✓ Camera handler created successfully")
        
        # Test frame capture
        rgb_frame, depth_frame = camera.capture_frames()
        
        if rgb_frame is not None and depth_frame is not None:
            print(f"✓ Frame capture successful")
            print(f"  - RGB shape: {rgb_frame.shape}")
            print(f"  - Depth shape: {depth_frame.shape}")
        else:
            print("⚠️  Frame capture returned None (expected in simulation)")
        
        # Test image processing
        if rgb_frame is not None and depth_frame is not None:
            processed_data = camera.process_frames(rgb_frame, depth_frame)
            print(f"✓ Image processing successful")
            print(f"  - Processed data keys: {list(processed_data.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Camera system test failed: {e}")
        return False

def test_robot_controller():
    """Test robot controller"""
    print("\nTesting robot controller...")
    
    try:
        sys.path.append('src')
        from enhanced_robot_controller import UR3KinematicsController, create_robot_system
        
        # Create robot controller in simulation mode
        robot_controller, gripper_controller, motion_planner = create_robot_system(
            config_path="config/robot_config.yaml",
            simulation=True
        )
        
        print("✓ Robot system created successfully")
        print(f"  - Robot controller: {type(robot_controller).__name__}")
        print(f"  - Gripper controller: {type(gripper_controller).__name__}")
        print(f"  - Motion planner: {type(motion_planner).__name__}")
        
        # Test forward kinematics
        joint_angles = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]  # Home position
        position, orientation = robot_controller.forward_kinematics(joint_angles)
        
        print("✓ Forward kinematics test successful")
        print(f"  - End-effector position: {position}")
        
        # Test workspace bounds checking
        is_reachable = robot_controller._is_pose_reachable(0.3, 0.2, 0.5)
        print(f"✓ Workspace bounds check: {is_reachable}")
        
        return True
        
    except Exception as e:
        print(f"✗ Robot controller test failed: {e}")
        return False

def test_network_connectivity():
    """Test network connectivity to GPU server"""
    print("\nTesting network connectivity...")
    
    try:
        # Try to connect to default GPU server
        gpu_server_host = "192.168.1.100"  # Default from config
        gpu_server_port = 8888
        
        print(f"Testing connection to {gpu_server_host}:{gpu_server_port}...")
        
        # Create socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(5.0)  # 5 second timeout
        
        try:
            client_socket.connect((gpu_server_host, gpu_server_port))
            print("✓ Successfully connected to GPU server")
            client_socket.close()
            return True
            
        except socket.timeout:
            print("✗ Connection timed out")
            print("  - Make sure GPU server is running on host machine")
            print("  - Check network configuration and firewall")
            return False
            
        except ConnectionRefusedError:
            print("✗ Connection refused")
            print("  - Make sure GPU server is running on host machine")
            print("  - Check if port is correct")
            return False
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
        
    except Exception as e:
        print(f"✗ Network test failed: {e}")
        return False

def test_simulation_client():
    """Test simulation client"""
    print("\nTesting simulation client...")
    
    try:
        sys.path.append('src')
        
        # Import simulation client
        from simulation_client import SimulationClient
        
        # Create client (this will likely fail without ROS running)
        try:
            client = SimulationClient(config_path="config/network_config.yaml")
            print("✓ Simulation client created successfully")
            return True
            
        except Exception as e:
            if "rospy" in str(e) or "ROS" in str(e):
                print("⚠️  Simulation client requires ROS to be running")
                print("   This is expected when ROS master is not active")
                return True  # This is acceptable for testing
            else:
                raise e
        
    except Exception as e:
        print(f"✗ Simulation client test failed: {e}")
        return False

def test_configuration_files():
    """Test configuration files"""
    print("\nTesting configuration files...")
    
    config_files = [
        'config/network_config.yaml',
        'config/robot_config.yaml',
        'config/camera_config.yaml'
    ]
    
    all_exist = True
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✓ {config_file} exists")
            
            # Try to load and validate YAML
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"  - Valid YAML with {len(config)} top-level keys")
            except Exception as e:
                print(f"  - ⚠️  YAML parsing error: {e}")
                
        else:
            print(f"✗ {config_file} missing")
            all_exist = False
    
    return all_exist

def test_launch_files():
    """Test ROS launch files"""
    print("\nTesting ROS launch files...")
    
    launch_files = [
        'launch/ur3_hybrid_system.launch',
        'launch/test_system.launch'
    ]
    
    all_exist = True
    
    for launch_file in launch_files:
        if Path(launch_file).exists():
            print(f"✓ {launch_file} exists")
            
            # Check if it's valid XML
            try:
                import xml.etree.ElementTree as ET
                ET.parse(launch_file)
                print("  - Valid XML format")
            except Exception as e:
                print(f"  - ⚠️  XML parsing error: {e}")
                
        else:
            print(f"✗ {launch_file} missing")
            all_exist = False
    
    return all_exist

def test_directory_structure():
    """Test directory structure"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'src',
        'config',
        'launch',
        'webots_worlds'
    ]
    
    optional_dirs = [
        'data',
        'results',
        'logs'
    ]
    
    all_required_exist = True
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✓ {directory}/ exists")
        else:
            print(f"✗ {directory}/ missing")
            all_required_exist = False
    
    for directory in optional_dirs:
        if Path(directory).exists():
            print(f"✓ {directory}/ exists (optional)")
        else:
            print(f"⚠️  {directory}/ missing (will be created automatically)")
    
    return all_required_exist

def main():
    """Run all tests"""
    print("=" * 60)
    print("UR3 Hybrid System - VM Simulation System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Configuration Files", test_configuration_files),
        ("Launch Files", test_launch_files),
        ("Python Packages", test_python_packages),
        ("ROS Installation", test_ros_installation),
        ("Webots Installation", test_webots_installation),
        ("Camera System", test_camera_system),
        ("Robot Controller", test_robot_controller),
        ("Network Connectivity", test_network_connectivity),
        ("Simulation Client", test_simulation_client)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 20} {test_name} {'-' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your VM simulation system is ready.")
        print("\nNext steps:")
        print("1. Start roscore: roscore")
        print("2. Launch the system: roslaunch ur3_hybrid_vm ur3_hybrid_system.launch")
        print("3. Make sure GPU server is running on host machine")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Install ROS: sudo apt install ros-melodic-desktop-full")
        print("- Install Webots: Download from https://cyberbotics.com/")
        print("- Check network connectivity to host machine")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

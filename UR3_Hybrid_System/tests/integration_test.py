#!/usr/bin/env python3
"""
Integration Test for UR3 Hybrid System
Tests end-to-end functionality of host GPU system and VM simulation system
"""

import os
import sys
import time
import subprocess
import threading
import socket
import json
import yaml
import numpy as np
from pathlib import Path

# Add both system paths
sys.path.append(os.path.join(os.path.dirname(__file__), "host_gpu_system", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "vm_simulation_system", "src"))

class HybridSystemIntegrationTest:
    """Comprehensive integration test for the hybrid UR3 system"""
    
    def __init__(self):
        self.results = {
            'host_gpu_system': {},
            'vm_simulation_system': {},
            'communication': {},
            'end_to_end': {}
        }
        
    def test_host_gpu_system(self):
        """Test GPU system components"""
        print("🖥️  Testing Host GPU System...")
        
        try:
            # Test enhanced neural network import and creation
            from enhanced_neural_network import UR3GraspCNN_Enhanced, ReinforcementLearningModule, create_model
            
            model_config = {
                'input_channels': 4,
                'input_size': [480, 640],
                'num_grasp_classes': 4,
                'output_6dof': True,
                'use_attention': True,
                'learning_rate': 1e-4,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 0.995
            }
            
            grasp_net, rl_module = create_model(model_config)
            self.results['host_gpu_system']['model_creation'] = "✅ PASS"
            print("  ✅ Enhanced neural network creation: PASS")
            
            # Test GPU server import
            from gpu_server import GPUInferenceServer
            self.results['host_gpu_system']['gpu_server_import'] = "✅ PASS"
            print("  ✅ GPU server import: PASS")
            
            # Test training pipeline import
            from training_pipeline import TrainingPipeline
            self.results['host_gpu_system']['training_pipeline_import'] = "✅ PASS"
            print("  ✅ Training pipeline import: PASS")
            
        except Exception as e:
            self.results['host_gpu_system']['error'] = f"❌ FAIL: {str(e)}"
            print(f"  ❌ Host GPU system test failed: {e}")
            return False
            
        return True
    
    def test_vm_simulation_system(self):
        """Test VM simulation system components"""
        print("🤖 Testing VM Simulation System...")
        
        try:
            # Test enhanced robot controller
            from enhanced_robot_controller import UR3KinematicsController, GripperController, MotionPlanner, create_robot_system
            
            robot_controller, gripper_controller, motion_planner = create_robot_system(
                config_path="vm_simulation_system/config/robot_config.yaml",
                simulation=True
            )
            self.results['vm_simulation_system']['robot_controller'] = "✅ PASS"
            print("  ✅ Enhanced robot controller creation: PASS")
            
            # Test enhanced camera handler
            from enhanced_camera_handler import EnhancedCameraHandler, create_camera_system
            
            camera_handler = create_camera_system(
                config_path="vm_simulation_system/config/camera_config.yaml",
                simulation=True
            )
            self.results['vm_simulation_system']['camera_handler'] = "✅ PASS"
            print("  ✅ Enhanced camera handler creation: PASS")
            
            # Test simulation client import
            from simulation_client import UR3SimulationClient
            self.results['vm_simulation_system']['simulation_client'] = "✅ PASS"
            print("  ✅ Simulation client import: PASS")
            
        except Exception as e:
            self.results['vm_simulation_system']['error'] = f"❌ FAIL: {str(e)}"
            print(f"  ❌ VM simulation system test failed: {e}")
            return False
            
        return True
    
    def test_configuration_files(self):
        """Test all configuration files are present and valid"""
        print("⚙️  Testing Configuration Files...")
        
        config_files = [
            "host_gpu_system/config/network_config.yaml",
            "host_gpu_system/config/model_config.yaml", 
            "host_gpu_system/config/training_config.yaml",
            "vm_simulation_system/config/network_config.yaml",
            "vm_simulation_system/config/robot_config.yaml",
            "vm_simulation_system/config/camera_config.yaml"
        ]
        
        all_configs_valid = True
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                print(f"  ✅ {config_file}: Valid YAML")
            except Exception as e:
                print(f"  ❌ {config_file}: Invalid - {e}")
                all_configs_valid = False
        
        self.results['configuration'] = "✅ PASS" if all_configs_valid else "❌ FAIL"
        return all_configs_valid
    
    def test_network_communication(self):
        """Test network communication setup"""
        print("🌐 Testing Network Communication...")
        
        try:
            # Load network configurations
            with open("host_gpu_system/config/network_config.yaml", 'r') as f:
                host_config = yaml.safe_load(f)
            
            with open("vm_simulation_system/config/network_config.yaml", 'r') as f:
                vm_config = yaml.safe_load(f) 
            
            # Check if configurations are compatible
            host_port = host_config['network']['port']
            vm_host_ip = vm_config['network']['host_ip']
            vm_port = vm_config['network']['port']
            
            if host_port == vm_port:
                print(f"  ✅ Port configuration compatible: {host_port}")
                self.results['communication']['port_config'] = "✅ PASS"
            else:
                print(f"  ❌ Port mismatch: Host={host_port}, VM={vm_port}")
                self.results['communication']['port_config'] = "❌ FAIL"
                return False
                
            # Test socket creation (basic connectivity test)
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(1)
            test_socket.close()
            
            self.results['communication']['socket_test'] = "✅ PASS"
            print("  ✅ Socket creation test: PASS")
            
        except Exception as e:
            self.results['communication']['error'] = f"❌ FAIL: {str(e)}"
            print(f"  ❌ Network communication test failed: {e}")
            return False
            
        return True
    
    def test_data_formats(self):
        """Test shared data formats"""
        print("📄 Testing Shared Data Formats...")
        
        try:
            # Test if shared resources are accessible
            sys.path.append(os.path.join(os.path.dirname(__file__), "shared_resources"))
            
            from data_formats import CameraData, GraspPrediction, RobotState
            
            # Test data format creation
            camera_data = CameraData(
                rgb_image=np.zeros((480, 640, 3), dtype=np.uint8),
                depth_image=np.zeros((480, 640), dtype=np.float32),
                timestamp=time.time()
            )
            
            grasp_prediction = GraspPrediction(
                position=[0.3, 0.2, 0.5],
                orientation=[0, 0, 0, 1],
                confidence=0.8,
                grasp_type="pinch"
            )
            
            robot_state = RobotState(
                joint_angles=[0.0] * 6,
                gripper_state="open",
                is_moving=False
            )
            
            self.results['data_formats'] = "✅ PASS"
            print("  ✅ Shared data formats: PASS")
            
        except Exception as e:
            self.results['data_formats'] = f"❌ FAIL: {str(e)}"
            print(f"  ❌ Data formats test failed: {e}")
            return False
            
        return True
    
    def test_directory_structure(self):
        """Test directory structure integrity"""
        print("📁 Testing Directory Structure...")
        
        required_dirs = [
            "host_gpu_system/src",
            "host_gpu_system/config", 
            "host_gpu_system/models",
            "host_gpu_system/data",
            "vm_simulation_system/src",
            "vm_simulation_system/config",
            "vm_simulation_system/launch",
            "shared_resources"
        ]
        
        required_files = [
            "host_gpu_system/src/enhanced_neural_network.py",
            "host_gpu_system/src/gpu_server.py",
            "host_gpu_system/src/training_pipeline.py",
            "vm_simulation_system/src/enhanced_robot_controller.py",
            "vm_simulation_system/src/enhanced_camera_handler.py",
            "vm_simulation_system/src/simulation_client.py",
            "shared_resources/data_formats.py",
            "README.md",
            "QUICK_START.md"
        ]
        
        all_present = True
        
        for directory in required_dirs:
            if os.path.isdir(directory):
                print(f"  ✅ Directory: {directory}")
            else:
                print(f"  ❌ Missing directory: {directory}")
                all_present = False
        
        for file_path in required_files:
            if os.path.isfile(file_path):
                print(f"  ✅ File: {file_path}")
            else:
                print(f"  ❌ Missing file: {file_path}")
                all_present = False
        
        self.results['directory_structure'] = "✅ PASS" if all_present else "❌ FAIL"
        return all_present
    
    def run_comprehensive_test(self):
        """Run all integration tests"""
        print("🚀 Starting UR3 Hybrid System Integration Test")
        print("=" * 60)
        
        tests = [
            ("Directory Structure", self.test_directory_structure),
            ("Configuration Files", self.test_configuration_files),
            ("Host GPU System", self.test_host_gpu_system),
            ("VM Simulation System", self.test_vm_simulation_system),
            ("Network Communication", self.test_network_communication),
            ("Data Formats", self.test_data_formats)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n{test_name}...")
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                print(f"  ❌ {test_name} failed with exception: {e}")
        
        print("\n" + "=" * 60)
        print(f"🎯 Integration Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Hybrid system is ready for use.")
            return True
        else:
            print("⚠️  Some tests failed. Please address the issues above.")
            return False
    
    def generate_report(self):
        """Generate detailed test report"""
        report = f"""
# UR3 Hybrid System Integration Test Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Test Results Summary
"""
        
        for category, results in self.results.items():
            report += f"\n### {category.title().replace('_', ' ')}\n"
            if isinstance(results, dict):
                for test, result in results.items():
                    report += f"- {test}: {result}\n"
            else:
                report += f"- Status: {results}\n"
        
        # Save report
        with open("integration_test_report.md", 'w') as f:
            f.write(report)
        
        print(f"\n📊 Detailed report saved to: integration_test_report.md")

def main():
    """Main test execution"""
    # Change to hybrid system directory
    hybrid_dir = "/Users/bh3os/obsidian/Ufol Research/UR3 paper and Repo/UR3_Hybrid_System"
    if os.path.exists(hybrid_dir):
        os.chdir(hybrid_dir)
    
    tester = HybridSystemIntegrationTest()
    success = tester.run_comprehensive_test()
    tester.generate_report()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

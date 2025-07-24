#!/usr/bin/env python3
"""
GPU Server for UR3 Deep Learning System
Runs on Windows host with RTX A6000 GPU
Handles neural network inference and communicates with VM simulation
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import socket
import json
import threading
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from enhanced_neural_network import UR3GraspCNN_Enhanced, ReinforcementLearningModule, create_model
from utils.logger import setup_logger
from utils.metrics import PerformanceMonitor
from data_pipeline import ImageProcessor

class GPUInferenceServer:
    """Main GPU server for handling VM requests and neural network inference"""
    
    def __init__(self, config_path: str = "config/network_config.yaml"):
        """Initialize the GPU server with configuration"""
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        self.logger = setup_logger("gpu_server", "data/logs/gpu_server.log")
        
        # Initialize device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Initialized GPU server on device: {self.device}")
        
        # Initialize neural network with enhanced model
        model_config = self._load_model_config()
        self.model, self.rl_module = create_model(model_config)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load pre-trained weights if available
        self._load_model_weights()
        
        # Initialize data processor
        self.image_processor = ImageProcessor(self.device)
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor()
        
        # Server state
        self.server_socket = None
        self.is_running = False
        self.client_connections = []
        
        self.logger.info("GPU server initialization complete")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                'network': {
                    'host_ip': '0.0.0.0',
                    'port': 8888,
                    'vm_ip': '192.168.1.100',
                    'timeout': 30,
                    'buffer_size': 4096
                }
            }
    
    def _load_model_config(self) -> Dict:
        """Load model configuration from YAML file"""
        try:
            with open("config/model_config.yaml", 'r') as f:
                config = yaml.safe_load(f)
                
            if config is None:
                config = {}
                
            model_config = config.get('model', {})
            training_config = config.get('training', {})
            
            # Add default values for enhanced model
            return {
                'input_channels': model_config.get('input_channels', 4),
                'input_size': model_config.get('image_size', [480, 640]),
                'num_grasp_classes': 4,
                'output_6dof': True,
                'use_attention': True,
                'learning_rate': training_config.get('learning_rate', 1e-4),
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 0.995
            }
        except FileNotFoundError:
            # Default enhanced model configuration
            return {
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
    
    def _load_model_weights(self):
        """Load pre-trained model weights"""
        model_path = Path("models/ur3_model.pth")
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load model weights: {e}")
        else:
            self.logger.warning("No pre-trained model found, using random weights")
    
    def preprocess_rgbd_data(self, rgbd_data: Dict) -> torch.Tensor:
        """Convert RGBD data from VM to tensor format for neural network"""
        try:
            # Extract RGB and depth data
            rgb = np.array(rgbd_data['rgb'], dtype=np.uint8).reshape((480, 640, 3))
            depth = np.array(rgbd_data['depth'], dtype=np.float32).reshape((480, 640))
            
            # Process using image processor
            rgbd_tensor = self.image_processor.process_rgbd(rgb, depth)
            
            return rgbd_tensor
            
        except Exception as e:
            self.logger.error(f"Error preprocessing RGBD data: {e}")
            raise
    
    def predict_grasp_pose(self, rgbd_tensor: torch.Tensor) -> np.ndarray:
        """Run neural network inference to predict grasp pose"""
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Forward pass through network
                prediction = self.model(rgbd_tensor)
                
                # Convert to numpy and remove batch dimension
                grasp_pose = prediction.cpu().numpy()[0]
                
                # Log inference time
                inference_time = time.time() - start_time
                self.performance_monitor.log_inference_time(inference_time)
                
                self.logger.debug(f"Inference completed in {inference_time:.3f}s")
                
                return grasp_pose
                
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            raise
    
    def handle_client_request(self, client_socket: socket.socket, address: Tuple[str, int]):
        """Handle incoming requests from VM client"""
        self.logger.info(f"New client connected from {address}")
        
        try:
            while self.is_running:
                # Receive message size
                size_data = client_socket.recv(4)
                if not size_data:
                    break
                
                message_size = int.from_bytes(size_data, byteorder='big')
                
                # Receive full message
                message_data = b''
                while len(message_data) < message_size:
                    chunk = client_socket.recv(min(message_size - len(message_data), 4096))
                    if not chunk:
                        break
                    message_data += chunk
                
                # Parse message
                message = json.loads(message_data.decode('utf-8'))
                
                # Handle different message types
                if message['type'] == 'camera_data':
                    response = self._handle_camera_data(message['data'])
                elif message['type'] == 'training_data':
                    response = self._handle_training_data(message['data'])
                elif message['type'] == 'ping':
                    response = {'type': 'pong', 'timestamp': time.time()}
                else:
                    response = {'type': 'error', 'message': 'Unknown message type'}
                
                # Send response
                response_data = json.dumps(response).encode('utf-8')
                client_socket.send(len(response_data).to_bytes(4, byteorder='big'))
                client_socket.send(response_data)
                
        except Exception as e:
            self.logger.error(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
            if client_socket in self.client_connections:
                self.client_connections.remove(client_socket)
            self.logger.info(f"Client {address} disconnected")
    
    def _handle_camera_data(self, camera_data: Dict) -> Dict:
        """Process camera data and return grasp prediction"""
        try:
            # Preprocess RGBD data
            rgbd_tensor = self.preprocess_rgbd_data(camera_data)
            
            # Predict grasp pose
            grasp_pose = self.predict_grasp_pose(rgbd_tensor)
            
            # Calculate confidence score
            confidence = float(np.mean(np.abs(grasp_pose)))
            
            # Log prediction
            self.logger.info(f"Grasp prediction: {grasp_pose[:3]} (confidence: {confidence:.3f})")
            
            return {
                'type': 'grasp_prediction',
                'pose': grasp_pose.tolist(),
                'confidence': confidence,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing camera data: {e}")
            return {
                'type': 'error',
                'message': str(e),
                'timestamp': time.time()
            }
    
    def _handle_training_data(self, training_data: Dict) -> Dict:
        """Handle training data from VM (for future model updates)"""
        try:
            # Store training sample for later use
            self._store_training_sample(training_data)
            
            return {
                'type': 'training_ack',
                'message': 'Training data received',
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error handling training data: {e}")
            return {
                'type': 'error',
                'message': str(e),
                'timestamp': time.time()
            }
    
    def _store_training_sample(self, training_data: Dict):
        """Store training sample for future model training"""
        # Implementation for storing training samples
        # This could save to a database or file for later training
        timestamp = int(time.time())
        
        # Save to data directory
        data_path = Path(f"data/training_samples_{timestamp}.json")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(data_path, 'w') as f:
            json.dump(training_data, f)
        
        self.logger.debug(f"Stored training sample: {data_path}")
    
    def start_server(self):
        """Start the GPU inference server"""
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind and listen
            host = self.config['network']['host_ip']
            port = self.config['network']['port']
            self.server_socket.bind((host, port))
            self.server_socket.listen(5)
            
            self.is_running = True
            self.logger.info(f"GPU server started on {host}:{port}")
            print(f"🔄 Server started, waiting for VM connections on {host}:{port}")
            
            # Accept connections
            while self.is_running:
                try:
                    client_socket, address = self.server_socket.accept()
                    self.client_connections.append(client_socket)
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client_request,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.is_running:
                        self.logger.error(f"Socket error: {e}")
        
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            raise
        finally:
            self.shutdown_server()
    
    def shutdown_server(self):
        """Gracefully shutdown the server"""
        self.logger.info("Shutting down GPU server...")
        self.is_running = False
        
        # Close client connections
        for client_socket in self.client_connections:
            try:
                client_socket.close()
            except:
                pass
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
        
        # Log performance statistics
        self.performance_monitor.log_summary()
        
        self.logger.info("GPU server shutdown complete")

def main():
    """Main entry point for GPU server"""
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️  CUDA not available, using CPU")
        print("   For optimal performance, ensure NVIDIA GPU drivers and CUDA are installed")
    
    # Initialize and start server
    try:
        server = GPUInferenceServer()
        print("🚀 GPU Server initialized successfully")
        server.start_server()
        
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise

if __name__ == "__main__":
    main()

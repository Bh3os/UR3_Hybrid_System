#!/usr/bin/env python3
"""
Enhanced Neural Network Architecture for UR3 Grasping System
Combines CNN-based grasp prediction with reinforcement learning
Based on the original drl_neural_ros implementation with improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict
from typing import Tuple, Optional, Dict, List
import logging

class UR3GraspCNN_Enhanced(nn.Module):
    """
    Enhanced Convolutional Neural Network for UR3 grasping prediction
    Supports both RGB and RGBD input, with multiple output heads
    """
    
    def __init__(self, 
                 input_channels: int = 4,  # RGBD by default
                 input_size: Tuple[int, int] = (480, 640),
                 num_grasp_classes: int = 4,
                 output_6dof: bool = True,
                 use_attention: bool = True):
        """
        Initialize the enhanced CNN architecture
        
        Args:
            input_channels: Number of input channels (3 for RGB, 4 for RGBD)
            input_size: Input image size (H, W)
            num_grasp_classes: Number of grasp classification classes
            output_6dof: Whether to output 6-DOF pose prediction
            use_attention: Whether to use attention mechanism
        """
        super(UR3GraspCNN_Enhanced, self).__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.num_grasp_classes = num_grasp_classes
        self.output_6dof = output_6dof
        self.use_attention = use_attention
        
        # Enhanced backbone with residual connections
        self.backbone = self._build_backbone()
        
        # Attention mechanism
        if self.use_attention:
            self.attention = SpatialAttention(512)
        
        # Calculate feature size
        self.feature_size = self._calculate_feature_size()
        
        # Multiple output heads
        self.grasp_classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_grasp_classes)
        )
        
        if self.output_6dof:
            self.pose_regressor = nn.Sequential(
                nn.Linear(self.feature_size, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 6)  # x, y, z, rx, ry, rz
            )
        
        # Quality prediction head
        self.quality_predictor = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Quality score between 0 and 1
        )
        
        self._initialize_weights()
    
    def _build_backbone(self) -> nn.Module:
        """Build the enhanced backbone network"""
        layers = []
        
        # Block 1
        layers.extend([
            nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        
        # Residual blocks
        layers.extend(self._make_layer(64, 64, 2, stride=1))   # Block 2
        layers.extend(self._make_layer(64, 128, 2, stride=2))  # Block 3
        layers.extend(self._make_layer(128, 256, 2, stride=2)) # Block 4
        layers.extend(self._make_layer(256, 512, 2, stride=2)) # Block 5
        
        # Global average pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        return nn.Sequential(*layers)
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int = 1) -> List[nn.Module]:
        """Create a residual layer"""
        layers = []
        
        # First block (potentially with stride)
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        
        return layers
    
    def _calculate_feature_size(self) -> int:
        """Calculate the feature size after backbone"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, *self.input_size)
            dummy_output = self.backbone(dummy_input)
            return dummy_output.view(dummy_output.size(0), -1).size(1)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, C, H, W)
            
        Returns:
            Dictionary containing different predictions
        """
        # Extract features
        features = self.backbone(x)
        
        # Apply attention if enabled
        if self.use_attention:
            features = self.attention(features)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        # Generate predictions
        outputs = {}
        
        # Grasp classification
        outputs['grasp_class'] = self.grasp_classifier(features)
        
        # 6-DOF pose regression
        if self.output_6dof:
            outputs['pose_6dof'] = self.pose_regressor(features)
        
        # Quality prediction
        outputs['quality'] = self.quality_predictor(features)
        
        return outputs


class ResidualBlock(nn.Module):
    """Residual block for improved gradient flow"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important regions"""
    
    def __init__(self, in_channels: int):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


class ReinforcementLearningModule(nn.Module):
    """
    Reinforcement learning module for training the grasp network
    Enhanced version with experience replay and improved exploration
    """
    
    def __init__(self, grasp_net: UR3GraspCNN_Enhanced, 
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        Initialize the RL module
        
        Args:
            grasp_net: The grasp prediction network
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
        """
        super(ReinforcementLearningModule, self).__init__()
        
        self.grasp_net = grasp_net
        self.target_net = self._create_target_network()
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Optimizers for different heads
        self.optimizers = {
            'grasp': torch.optim.Adam(
                list(grasp_net.backbone.parameters()) + 
                list(grasp_net.grasp_classifier.parameters()), 
                lr=learning_rate
            ),
            'pose': torch.optim.Adam(
                list(grasp_net.backbone.parameters()) + 
                list(grasp_net.pose_regressor.parameters()) if grasp_net.output_6dof else [], 
                lr=learning_rate
            ),
            'quality': torch.optim.Adam(
                list(grasp_net.backbone.parameters()) + 
                list(grasp_net.quality_predictor.parameters()), 
                lr=learning_rate
            )
        }
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.quality_loss = nn.BCELoss()
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0.0,
            'success_rate': 0.0,
            'losses': {'grasp': [], 'pose': [], 'quality': []}
        }
    
    def _create_target_network(self) -> UR3GraspCNN_Enhanced:
        """Create target network for stable training"""
        target_net = UR3GraspCNN_Enhanced(
            input_channels=self.grasp_net.input_channels,
            input_size=self.grasp_net.input_size,
            num_grasp_classes=self.grasp_net.num_grasp_classes,
            output_6dof=self.grasp_net.output_6dof,
            use_attention=self.grasp_net.use_attention
        )
        target_net.load_state_dict(self.grasp_net.state_dict())
        target_net.eval()
        return target_net
    
    def select_action(self, state: torch.Tensor, 
                     training: bool = True) -> Tuple[int, Dict[str, float]]:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state tensor
            training: Whether in training mode
            
        Returns:
            Tuple of (action, prediction_dict)
        """
        if training and np.random.random() < self.epsilon:
            # Random action for exploration
            action = np.random.randint(0, self.grasp_net.num_grasp_classes)
            predictions = {'confidence': 0.0}
        else:
            # Greedy action
            with torch.no_grad():
                outputs = self.grasp_net(state)
                grasp_probs = F.softmax(outputs['grasp_class'], dim=1)
                action = torch.argmax(grasp_probs, dim=1).item()
                
                predictions = {
                    'confidence': torch.max(grasp_probs).item(),
                    'quality': outputs['quality'].item()
                }
                
                if 'pose_6dof' in outputs:
                    predictions['pose'] = outputs['pose_6dof'].cpu().numpy().flatten()
        
        # Decay epsilon
        if training:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return action, predictions
    
    def update_networks(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update networks using batch of experience data
        
        Args:
            batch_data: Dictionary containing batch data
            
        Returns:
            Dictionary of loss values
        """
        losses = {}
        
        states = batch_data['states']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        next_states = batch_data['next_states']
        dones = batch_data['dones']
        
        # Get current predictions
        current_outputs = self.grasp_net(states)
        
        # Update grasp classifier
        if 'grasp_labels' in batch_data:
            grasp_loss = self.classification_loss(
                current_outputs['grasp_class'], 
                batch_data['grasp_labels']
            )
            
            self.optimizers['grasp'].zero_grad()
            grasp_loss.backward(retain_graph=True)
            self.optimizers['grasp'].step()
            
            losses['grasp'] = grasp_loss.item()
        
        # Update pose regressor
        if self.grasp_net.output_6dof and 'pose_labels' in batch_data:
            pose_loss = self.regression_loss(
                current_outputs['pose_6dof'], 
                batch_data['pose_labels']
            )
            
            self.optimizers['pose'].zero_grad()
            pose_loss.backward(retain_graph=True)
            self.optimizers['pose'].step()
            
            losses['pose'] = pose_loss.item()
        
        # Update quality predictor using TD learning
        with torch.no_grad():
            next_outputs = self.target_net(next_states)
            target_quality = rewards + self.gamma * next_outputs['quality'].squeeze() * (1 - dones)
        
        quality_loss = self.quality_loss(
            current_outputs['quality'].squeeze(), 
            target_quality
        )
        
        self.optimizers['quality'].zero_grad()
        quality_loss.backward()
        self.optimizers['quality'].step()
        
        losses['quality'] = quality_loss.item()
        
        # Update training statistics
        for key, value in losses.items():
            self.training_stats['losses'][key].append(value)
        
        return losses
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_net.load_state_dict(self.grasp_net.state_dict())
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.grasp_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizers': {key: opt.state_dict() for key, opt in self.optimizers.items()},
            'training_stats': self.training_stats,
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.grasp_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        
        for key, opt in self.optimizers.items():
            if key in checkpoint['optimizers']:
                opt.load_state_dict(checkpoint['optimizers'][key])
        
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)


# Utility functions for data processing
class ImageProcessor:
    """Enhanced image processing utilities"""
    
    def __init__(self, device: torch.device = None, image_size: Tuple[int, int] = (480, 640)):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # Define transforms
        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], 
                               std=[0.229, 0.224, 0.225, 0.5])  # RGBD normalization
        ])
    
    def process_rgbd_image(self, rgb_image: np.ndarray, 
                          depth_image: np.ndarray) -> torch.Tensor:
        """
        Process RGB and depth images into a single tensor
        
        Args:
            rgb_image: RGB image array (H, W, 3)
            depth_image: Depth image array (H, W)
            
        Returns:
            Processed RGBD tensor (1, 4, H, W)
        """
        # Normalize depth image
        depth_normalized = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
        
        # Combine RGB and D
        rgbd_image = np.concatenate([rgb_image, depth_normalized[:, :, np.newaxis]], axis=2)
        
        # Convert to tensor and apply transforms
        rgbd_tensor = torch.from_numpy(rgbd_image.transpose(2, 0, 1)).float()
        rgbd_tensor = rgbd_tensor.unsqueeze(0).to(self.device)
        
        return rgbd_tensor
    
    def augment_image(self, image: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation"""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            image = torch.flip(image, [3])
        
        # Random rotation
        angle = torch.randint(-15, 15, (1,)).item()
        image = transforms.functional.rotate(image, angle)
        
        # Random brightness and contrast
        brightness_factor = torch.uniform(0.8, 1.2).item()
        contrast_factor = torch.uniform(0.8, 1.2).item()
        
        image = transforms.functional.adjust_brightness(image, brightness_factor)
        image = transforms.functional.adjust_contrast(image, contrast_factor)
        
        return image


def create_model(config: Dict) -> Tuple[UR3GraspCNN_Enhanced, ReinforcementLearningModule]:
    """
    Factory function to create model based on configuration
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Tuple of (grasp_network, rl_module)
    """
    grasp_net = UR3GraspCNN_Enhanced(
        input_channels=config.get('input_channels', 4),
        input_size=tuple(config.get('input_size', [480, 640])),
        num_grasp_classes=config.get('num_grasp_classes', 4),
        output_6dof=config.get('output_6dof', True),
        use_attention=config.get('use_attention', True)
    )
    
    rl_module = ReinforcementLearningModule(
        grasp_net=grasp_net,
        learning_rate=config.get('learning_rate', 1e-4),
        gamma=config.get('gamma', 0.99),
        epsilon_start=config.get('epsilon_start', 1.0),
        epsilon_end=config.get('epsilon_end', 0.01),
        epsilon_decay=config.get('epsilon_decay', 0.995)
    )
    
    return grasp_net, rl_module

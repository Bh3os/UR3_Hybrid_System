#!/usr/bin/env python3
"""
Data Pipeline for Host GPU System
Handles image processing, batching, and tensor operations
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple
import torchvision.transforms as transforms

class ImageProcessor:
    """Process RGBD images for neural network input"""
    
    def __init__(self, device: torch.device, input_size: Tuple[int, int] = (480, 640)):
        self.device = device
        self.input_size = input_size
        
        # Image preprocessing transforms
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def process_rgbd(self, rgb: np.ndarray, depth: np.ndarray) -> torch.Tensor:
        """Process RGB and depth images into tensor format"""
        
        # Ensure correct shapes
        if rgb.shape[:2] != self.input_size:
            rgb = cv2.resize(rgb, (self.input_size[1], self.input_size[0]))
        
        if depth.shape != self.input_size:
            depth = cv2.resize(depth, (self.input_size[1], self.input_size[0]))
        
        # Normalize RGB to [0, 1]
        rgb_normalized = rgb.astype(np.float32) / 255.0
        
        # Normalize depth (clip to reasonable range)
        depth_normalized = np.clip(depth / 2000.0, 0, 1).astype(np.float32)
        
        # Add depth as 4th channel
        rgbd = np.concatenate([rgb_normalized, depth_normalized[..., np.newaxis]], axis=2)
        
        # Convert to tensor (H, W, C) -> (C, H, W) and add batch dimension
        rgbd_tensor = torch.from_numpy(rgbd).permute(2, 0, 1).unsqueeze(0)
        
        return rgbd_tensor.to(self.device)
    
    def batch_process(self, rgbd_list: List[Tuple[np.ndarray, np.ndarray]]) -> torch.Tensor:
        """Process multiple RGBD images in batch"""
        batch_tensors = []
        
        for rgb, depth in rgbd_list:
            tensor = self.process_rgbd(rgb, depth)
            batch_tensors.append(tensor)
        
        return torch.cat(batch_tensors, dim=0)

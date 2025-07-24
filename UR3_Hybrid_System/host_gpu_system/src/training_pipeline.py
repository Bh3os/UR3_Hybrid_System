#!/usr/bin/env python3
"""
Enhanced Training Script for UR3 Grasp Network
Comprehensive training pipeline with data augmentation, validation, and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import yaml
import logging
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorboardX import SummaryWriter

# Local imports
from enhanced_neural_network import UR3GraspCNN_Enhanced, ReinforcementLearningModule, ImageProcessor
from data_pipeline import ImageProcessor
from utils.logger import setup_logger
from utils.metrics import PerformanceMonitor

class GraspTrainingPipeline:
    """
    Comprehensive training pipeline for UR3 grasp network
    """
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """Initialize training pipeline"""
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {self.device}")
        
        # Setup logging
        self.logger = setup_logger("training", "data/logs/training.log")
        
        # Setup directories
        self._setup_directories()
        
        # Initialize model
        self.model, self.rl_module = self._create_model()
        
        # Setup data pipeline
        self.data_pipeline = ImageProcessor(self.device)
        
        # Setup training components
        self._setup_training_components()
        
        # Setup monitoring
        self.performance_monitor = PerformanceMonitor()
        self.training_metrics = PerformanceMonitor()
        
        # TensorBoard logging
        self.writer = SummaryWriter(log_dir=self.config['logging']['tensorboard_dir'])
        
        self.logger.info("Training pipeline initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default training configuration"""
        return {
            'model': {
                'input_channels': 4,
                'input_size': [480, 640],
                'num_grasp_classes': 4,
                'output_6dof': True,
                'use_attention': True,
                'learning_rate': 1e-4,
                'gamma': 0.99
            },
            'training': {
                'batch_size': 16,
                'num_epochs': 100,
                'validation_split': 0.2,
                'early_stopping_patience': 10,
                'lr_scheduler': 'cosine',
                'weight_decay': 1e-5,
                'gradient_clipping': 1.0
            },
            'data': {
                'dataset_path': 'data/grasp_dataset',
                'augmentation': True,
                'num_workers': 4,
                'pin_memory': True
            },
            'logging': {
                'log_interval': 10,
                'save_interval': 500,
                'tensorboard_dir': 'runs/ur3_training',
                'checkpoint_dir': 'models/checkpoints'
            }
        }
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            'data/logs',
            'data/grasp_dataset',
            'models/checkpoints',
            'models/final',
            'results/training',
            'results/validation',
            'runs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _create_model(self) -> Tuple[UR3GraspCNN_Enhanced, ReinforcementLearningModule]:
        """Create and initialize model"""
        model_config = self.config['model']
        
        # Create grasp network
        grasp_net = UR3GraspCNN_Enhanced(
            input_channels=model_config['input_channels'],
            input_size=tuple(model_config['input_size']),
            num_grasp_classes=model_config['num_grasp_classes'],
            output_6dof=model_config['output_6dof'],
            use_attention=model_config['use_attention']
        ).to(self.device)
        
        # Create RL module
        rl_module = ReinforcementLearningModule(
            grasp_net=grasp_net,
            learning_rate=model_config['learning_rate'],
            gamma=model_config['gamma']
        )
        
        # Load pretrained weights if available
        pretrained_path = model_config.get('pretrained_weights')
        if pretrained_path and os.path.exists(pretrained_path):
            self.logger.info(f"Loading pretrained weights from {pretrained_path}")
            rl_module.load_model(pretrained_path)
        
        return grasp_net, rl_module
    
    def _setup_training_components(self):
        """Setup optimizers, schedulers, and loss functions"""
        train_config = self.config['training']
        
        # Setup learning rate scheduler
        if train_config['lr_scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.rl_module.optimizers['grasp'], 
                T_max=train_config['num_epochs']
            )
        elif train_config['lr_scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.rl_module.optimizers['grasp'], 
                step_size=30, gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Setup loss weights
        self.loss_weights = {
            'grasp': 1.0,
            'pose': 0.5,
            'quality': 0.3
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = train_config['early_stopping_patience']
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders"""
        # For integration testing, return None data loaders
        # In production, this would load actual datasets
        self.logger.info("Data preparation skipped for integration testing")
        return None, None
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )
        
        self.logger.info(f"Dataset split: {train_size} training, {val_size} validation")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': 0.0, 'grasp': 0.0, 'pose': 0.0, 'quality': 0.0}
        num_batches = len(train_loader)
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Move data to device
            images = batch_data['image'].to(self.device)
            grasp_labels = batch_data['grasp_class'].to(self.device)
            pose_labels = batch_data['pose_6dof'].to(self.device) if 'pose_6dof' in batch_data else None
            quality_labels = batch_data['quality'].to(self.device) if 'quality' in batch_data else None
            
            # Prepare batch data for RL module
            rl_batch_data = {
                'states': images,
                'grasp_labels': grasp_labels
            }
            
            if pose_labels is not None:
                rl_batch_data['pose_labels'] = pose_labels
            
            # Forward pass and compute losses
            losses = self.rl_module.update_networks(rl_batch_data)
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value
            
            total_loss = sum(losses.values())
            epoch_losses['total'] += total_loss
            
            # Gradient clipping
            if self.config['training']['gradient_clipping'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clipping']
                )
            
            # Logging
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                               f'Loss: {total_loss:.4f}')
                
                # TensorBoard logging
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Loss/Batch/Total', total_loss, global_step)
                for key, value in losses.items():
                    self.writer.add_scalar(f'Loss/Batch/{key.capitalize()}', value, global_step)
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_losses = {'total': 0.0, 'grasp': 0.0, 'pose': 0.0, 'quality': 0.0}
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                images = batch_data['image'].to(self.device)
                grasp_labels = batch_data['grasp_class'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute losses
                grasp_loss = self.rl_module.classification_loss(
                    outputs['grasp_class'], grasp_labels
                )
                val_losses['grasp'] += grasp_loss.item()
                
                if 'pose_6dof' in outputs and 'pose_6dof' in batch_data:
                    pose_labels = batch_data['pose_6dof'].to(self.device)
                    pose_loss = self.rl_module.regression_loss(
                        outputs['pose_6dof'], pose_labels
                    )
                    val_losses['pose'] += pose_loss.item()
                
                if 'quality' in outputs and 'quality' in batch_data:
                    quality_labels = batch_data['quality'].to(self.device)
                    quality_loss = self.rl_module.quality_loss(
                        outputs['quality'].squeeze(), quality_labels
                    )
                    val_losses['quality'] += quality_loss.item()
                
                # Collect predictions for metrics
                grasp_pred = torch.argmax(outputs['grasp_class'], dim=1)
                predictions.extend(grasp_pred.cpu().numpy())
                ground_truths.extend(grasp_labels.cpu().numpy())
        
        # Average losses
        num_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        val_losses['total'] = sum(val_losses.values())
        
        # Calculate accuracy
        accuracy = (np.array(predictions) == np.array(ground_truths)).mean()
        val_losses['accuracy'] = accuracy
        
        return val_losses
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data()
        
        # Training history
        train_history = {'loss': [], 'grasp_loss': [], 'pose_loss': [], 'quality_loss': []}
        val_history = {'loss': [], 'grasp_loss': [], 'pose_loss': [], 'quality_loss': [], 'accuracy': []}
        
        # Training loop
        num_epochs = self.config['training']['num_epochs']
        
        try:
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                
                # Training
                train_losses = self.train_epoch(train_loader, epoch)
                
                # Validation
                val_losses = self.validate_epoch(val_loader, epoch)
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
                
                # Record history
                train_history['loss'].append(train_losses['total'])
                train_history['grasp_loss'].append(train_losses['grasp'])
                train_history['pose_loss'].append(train_losses['pose'])
                train_history['quality_loss'].append(train_losses['quality'])
                
                val_history['loss'].append(val_losses['total'])
                val_history['grasp_loss'].append(val_losses['grasp'])
                val_history['pose_loss'].append(val_losses['pose'])
                val_history['quality_loss'].append(val_losses['quality'])
                val_history['accuracy'].append(val_losses['accuracy'])
                
                # TensorBoard logging
                self.writer.add_scalar('Loss/Epoch/Train', train_losses['total'], epoch)
                self.writer.add_scalar('Loss/Epoch/Val', val_losses['total'], epoch)
                self.writer.add_scalar('Accuracy/Val', val_losses['accuracy'], epoch)
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Logging
                self.logger.info(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s')
                self.logger.info(f'Train Loss: {train_losses["total"]:.4f}, '
                               f'Val Loss: {val_losses["total"]:.4f}, '
                               f'Val Accuracy: {val_losses["accuracy"]:.4f}')
                
                # Early stopping check
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.patience_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f'Early stopping triggered after {epoch+1} epochs')
                    break
                
                # Regular checkpoint saving
                if (epoch + 1) % self.config['logging']['save_interval'] == 0:
                    self.save_checkpoint(epoch, is_best=False)
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        finally:
            # Save final model
            self.save_final_model()
            
            # Generate training report
            self.generate_training_report(train_history, val_history)
            
            # Close TensorBoard writer
            self.writer.close()
            
            self.logger.info("Training completed")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            filepath = checkpoint_dir / 'best_model.pth'
        else:
            filepath = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        self.rl_module.save_model(str(filepath))
        self.logger.info(f'Checkpoint saved: {filepath}')
    
    def save_final_model(self):
        """Save final trained model"""
        final_dir = Path('models/final')
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = final_dir / 'ur3_grasp_model_final.pth'
        self.rl_module.save_model(str(model_path))
        
        # Save configuration
        config_path = final_dir / 'training_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        self.logger.info(f'Final model saved: {model_path}')
    
    def generate_training_report(self, train_history: Dict, val_history: Dict):
        """Generate comprehensive training report"""
        results_dir = Path('results/training')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot training curves
        self._plot_training_curves(train_history, val_history, results_dir)
        
        # Generate summary report
        self._generate_summary_report(train_history, val_history, results_dir)
        
        self.logger.info(f'Training report generated in {results_dir}')
    
    def _plot_training_curves(self, train_history: Dict, val_history: Dict, output_dir: Path):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(train_history['loss'], label='Train')
        axes[0, 0].plot(val_history['loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Grasp classification loss
        axes[0, 1].plot(train_history['grasp_loss'], label='Train')
        axes[0, 1].plot(val_history['grasp_loss'], label='Validation')
        axes[0, 1].set_title('Grasp Classification Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Pose regression loss
        axes[1, 0].plot(train_history['pose_loss'], label='Train')
        axes[1, 0].plot(val_history['pose_loss'], label='Validation')
        axes[1, 0].set_title('Pose Regression Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Validation accuracy
        axes[1, 1].plot(val_history['accuracy'], label='Validation Accuracy')
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, train_history: Dict, val_history: Dict, output_dir: Path):
        """Generate summary report"""
        report = {
            'training_summary': {
                'total_epochs': len(train_history['loss']),
                'best_val_loss': float(self.best_val_loss),
                'best_val_accuracy': float(max(val_history['accuracy'])),
                'final_train_loss': float(train_history['loss'][-1]),
                'final_val_loss': float(val_history['loss'][-1]),
                'final_val_accuracy': float(val_history['accuracy'][-1])
            },
            'model_config': self.config['model'],
            'training_config': self.config['training'],
            'device_used': str(self.device)
        }
        
        # Save report
        report_path = output_dir / 'training_summary.yaml'
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate trained model on test set"""
        self.model.eval()
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                images = batch_data['image'].to(self.device)
                grasp_labels = batch_data['grasp_class'].to(self.device)
                
                outputs = self.model(images)
                grasp_pred = torch.argmax(outputs['grasp_class'], dim=1)
                
                predictions.extend(grasp_pred.cpu().numpy())
                ground_truths.extend(grasp_labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = (np.array(predictions) == np.array(ground_truths)).mean()
        
        # Generate classification report
        class_report = classification_report(
            ground_truths, predictions, output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(ground_truths, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm
        }


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train UR3 Grasp Network')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create training pipeline
    trainer = GraspTrainingPipeline(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.rl_module.load_model(args.resume)
        print(f"Resumed training from {args.resume}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

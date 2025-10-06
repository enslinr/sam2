"""
General-purpose visualization utilities for multi-class segmentation.
Provides functions to create IoU visualizations from one-hot encoded masks.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import cv2
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class TrainingVisualizer:
    """General-purpose class for creating segmentation visualizations."""
    
    def __init__(self, device: str = "cuda", include_background_as_class: bool = False):
        self.device = device
        self.include_background_as_class = include_background_as_class
        # Define a colormap for different classes
        self.class_colors = [
            '#000000',  # Black - Background, Class 0'
            '#FF0000',  # Red - NFL
            '#C6E0B4',  # Green - GCL
            '#0F5AF1',  # Blue - IPL
            '#FFFF00',  # Yellow - INL
            '#548235',  # Green - OPL
            '#49E838',  # Green - ONL
            '#00B0F0',  # Blue - IS
            '#C65911',  # Brown - OS
            '#FF00FF',  # Pink - RPE
            '#2F75B5',  # Blue - Choroid
            '#F4B084',  # Brown - Optic disc
            '#FFC000',  # Orange - Fluid
        ]
    
    def _hex_to_rgb01(self, hex_color: str) -> tuple:
        """'#RRGGBB' -> (r,g,b) in [0,1]."""
        return mcolors.to_rgb(hex_color)

    def index_mask_to_rgb(self, idx_mask: torch.Tensor, palette: list) -> np.ndarray:
        """
        Convert a [H,W] index mask to an RGB image using a hex palette (list of '#RRGGBB').
        If an index exceeds palette length, it is modulo-wrapped.
        """
        arr = idx_mask.cpu().numpy().astype(np.int32)
        H, W = arr.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        # Pre-convert palette to floats
        pal = [np.array(self._hex_to_rgb01(c), dtype=np.float32) for c in palette]
        L = len(pal)
        # Vectorized assignment
        for i in np.unique(arr):
            rgb[arr == i] = pal[i % L]
        return rgb

    def compute_iou_per_class(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> Dict[str, float]:
        """
        Compute IoU for each class individually.
        
        Args:
            pred_masks: torch.Tensor of shape [num_classes, H, W] - predicted one-hot masks
            target_masks: torch.Tensor of shape [num_classes, H, W] - target one-hot masks
        
        Returns:
            Dict mapping class names to IoU scores
        """
        iou_scores = {}
        
        for class_idx in range(pred_masks.shape[0]):
            pred_class = pred_masks[class_idx]
            target_class = target_masks[class_idx]
            
            # Compute IoU for this class
            intersection = (pred_class * target_class).sum().item()
            union = pred_class.sum().item() + target_class.sum().item() - intersection
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 1.0 if intersection == 0 else 0.0
            
            iou_scores[f'class_{class_idx}'] = iou
        
        return iou_scores
    
    def create_colored_mask_overlay(self, masks: torch.Tensor, alpha: float = 0.6) -> np.ndarray:
        """
        Create a colored overlay from multiple one-hot binary masks.
        
        Args:
            masks: torch.Tensor of shape [num_classes, H, W] - one-hot binary masks
            alpha: transparency for the overlay
        
        Returns:
            np.ndarray of shape [H, W, 3] - colored mask overlay
        """
        H, W = masks.shape[1], masks.shape[2]
        overlay = np.zeros((H, W, 3), dtype=np.float32)
        
        for class_idx in range(masks.shape[0]):
            mask = masks[class_idx].cpu().numpy()
            color = np.array(mcolors.to_rgb(self.class_colors[class_idx % len(self.class_colors)]))
            
            # Add color where mask is active
            for c in range(3):
                overlay[:, :, c] += mask * color[c] * alpha
        
        return np.clip(overlay, 0, 1)
    
    def create_iou_visualization(
        self, 
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        title: str = "",
        epoch: Optional[int] = None
    ) -> Tuple[plt.Figure, Dict[str, float]]:
        """
        Create IoU visualization comparing predicted vs target masks.
        
        Args:
            pred_masks: torch.Tensor of shape [num_classes, H, W] - predicted one-hot masks
            target_masks: torch.Tensor of shape [num_classes, H, W] - target one-hot masks
            title: str - title for the visualization
            epoch: Optional[int] - current epoch (for display)
        
        Returns:
            Tuple of (matplotlib Figure, IoU scores dict)
        """
        # Ensure masks are on CPU and binary
        pred_masks = (pred_masks.cpu() > 0.5).float()
        target_masks = (target_masks.cpu() > 0.5).float()
        
        # Compute IoU per class
        iou_scores = self.compute_iou_per_class(pred_masks, target_masks)
        
        # Create colored overlays
        pred_overlay = self.create_colored_mask_overlay(pred_masks)
        target_overlay = self.create_colored_mask_overlay(target_masks)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot predicted masks
        axes[0].imshow(pred_overlay)
        pred_title = 'Predicted Masks'
        if epoch is not None:
            pred_title += f'\nEpoch {epoch}'
        axes[0].set_title(pred_title, fontsize=12)
        axes[0].axis('off')
        
        # Plot target masks
        axes[1].imshow(target_overlay)
        axes[1].set_title('Ground Truth Masks', fontsize=12)
        axes[1].axis('off')
        
        # Plot IoU scores
        classes = list(iou_scores.keys())
        ious = list(iou_scores.values())
        
        bars = axes[2].bar(classes, ious, color=self.class_colors[:len(classes)])
        axes[2].set_title('IoU Scores per Class', fontsize=12)
        axes[2].set_ylabel('IoU Score')
        axes[2].set_ylim(0, 1.0)
        axes[2].tick_params(axis='x', rotation=45)
        
        # Add IoU values on bars
        for bar, iou in zip(bars, ious):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{iou:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add overall information
        mean_iou = np.mean(ious) if ious else 0.0
        suptitle = title if title else 'Segmentation Comparison'
        fig.suptitle(f'{suptitle}\nMean IoU: {mean_iou:.3f}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, iou_scores
    
    def create_class_legend(self, num_classes: int) -> plt.Figure:
        """
        Create a legend showing class colors.
        
        Args:
            num_classes: number of classes to show
        
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 2))
        
        # Create legend patches
        legend_elements = []
        for i in range(num_classes):
            color = self.class_colors[i % len(self.class_colors)]
            patch = patches.Patch(color=color, label=f'Class {i + 1}')
            legend_elements.append(patch)
        
        ax.legend(handles=legend_elements, loc='center', ncol=num_classes)
        ax.axis('off')
        ax.set_title('Class Color Legend', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig


# Conversion utilities for different mask formats
class MaskConverter:
    """Utility class for converting between different mask formats."""
    
    @staticmethod
    def logits_to_one_hot(logits: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        Convert logits to non-overlapping one-hot masks using argmax.
        
        Args:
            logits: torch.Tensor of shape [num_classes, H, W] or [num_classes, 1, H, W]
            threshold: minimum logit value for assignment
            
        Returns:
            torch.Tensor of shape [num_classes, H, W] - one-hot masks
        """
        if logits.dim() == 4:
            logits = logits.squeeze(1)  # Remove channel dim if present
        
        # Find max class per pixel
        max_values, max_indices = torch.max(logits, dim=0)
        above_threshold = max_values > threshold
        
        # Create one-hot masks
        num_classes = logits.shape[0]
        one_hot_masks = torch.zeros_like(logits)
        
        for class_idx in range(num_classes):
            class_assignment = (max_indices == class_idx) & above_threshold
            one_hot_masks[class_idx] = class_assignment.float()
            
        return one_hot_masks
    
    @staticmethod
    def probabilities_to_one_hot(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Convert probability masks to one-hot using thresholding or argmax.
        
        Args:
            probs: torch.Tensor of shape [num_classes, H, W] - probability masks
            threshold: threshold for binary conversion (use argmax if None)
            
        Returns:
            torch.Tensor of shape [num_classes, H, W] - one-hot masks
        """
        if threshold is None:
            # Use argmax for mutually exclusive classes
            max_indices = torch.argmax(probs, dim=0)
            num_classes = probs.shape[0]
            one_hot_masks = torch.zeros_like(probs)
            
            for class_idx in range(num_classes):
                one_hot_masks[class_idx] = (max_indices == class_idx).float()
        else:
            # Use threshold for potentially overlapping classes
            one_hot_masks = (probs > threshold).float()
            
        return one_hot_masks
    
    @staticmethod
    def class_indices_to_one_hot(class_mask: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Convert class index mask to one-hot format.
        
        Args:
            class_mask: torch.Tensor of shape [H, W] - class indices (1-based)
            num_classes: number of classes
            
        Returns:
            torch.Tensor of shape [num_classes, H, W] - one-hot masks
        """
        one_hot_masks = torch.zeros(num_classes, *class_mask.shape, dtype=torch.float32)
        
        for class_idx in range(num_classes):
            one_hot_masks[class_idx] = (class_mask == (class_idx)).float()
            
        return one_hot_masks
    



def create_training_visualizer(device: str = "cuda", include_background_as_class: bool = False) -> TrainingVisualizer:
    """
    Factory function to create a TrainingVisualizer instance.
    
    Args:
        device: device to use for computations
    
    Returns:
        TrainingVisualizer instance
    """
    return TrainingVisualizer(device=device, include_background_as_class=include_background_as_class)


# Example usage functions
def visualize_from_logits(
    pred_logits: torch.Tensor,
    target_class_mask: torch.Tensor,
    visualizer: TrainingVisualizer,
    title: str = "",
    epoch: Optional[int] = None,
    threshold: float = 0.0
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Create visualization from raw logits and class index mask.
    
    Args:
        pred_logits: torch.Tensor of shape [num_classes, H, W] - raw logits
        target_class_mask: torch.Tensor of shape [H, W] - class indices (1-based)
        visualizer: TrainingVisualizer instance
        title: title for visualization
        epoch: current epoch
        threshold: threshold for logit conversion
        
    Returns:
        Tuple of (figure, iou_scores)
    """
    # Convert logits to one-hot
    pred_one_hot = MaskConverter.logits_to_one_hot(pred_logits, threshold)
    
    # Convert target to one-hot
    num_classes = pred_logits.shape[0]
    target_one_hot = MaskConverter.class_indices_to_one_hot(target_class_mask, num_classes)
    
    return visualizer.create_iou_visualization(pred_one_hot, target_one_hot, title, epoch)


def visualize_from_probabilities(
    pred_probs: torch.Tensor,
    target_mask: torch.Tensor,
    visualizer: TrainingVisualizer,
    title: str = "",
    epoch: Optional[int] = None,
    use_argmax: bool = True
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Create visualization from probability masks and class index mask.
    
    Args:
        pred_probs: torch.Tensor of shape [num_classes, H, W] - probability masks
        target_mask: torch.Tensor of shape [H, W] - class indices
        visualizer: TrainingVisualizer instance
        title: title for visualization
        epoch: current epoch
        use_argmax: whether to use argmax (True) or threshold (False)
        
    Returns:
        Tuple of (figure, iou_scores)
    """
    # Convert probabilities to one-hot
    threshold = None if use_argmax else 0.5
    pred_one_hot = MaskConverter.probabilities_to_one_hot(pred_probs, threshold)
    
    # Convert target to one-hot
    num_classes = pred_probs.shape[0]
    target_one_hot = MaskConverter.class_indices_to_one_hot(target_mask, num_classes)
    
    return visualizer.create_iou_visualization(pred_one_hot, target_one_hot, title, epoch)
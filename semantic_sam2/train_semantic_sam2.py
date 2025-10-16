#!/usr/bin/env python3
"""
Training script for Semantic SAM2 on paired image/mask segmentation datasets.

This script expects paired directories of images and masks for training and
validation. Image files are matched to masks via their filename stem. Masks
should encode class indices (0..K-1) and may optionally include an ignore
index value (e.g., 255).
"""

import argparse
import logging
import random
import inspect
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import albumentations as A

from setup_imports import root_dir
from semantic_sam2.build_semantic_sam2 import build_semantic_sam2
from semantic_sam2.training_utils import FocalLossCEAligned
from semantic_sam2.wandb_wrapper import create_wandb_wrapper
from semantic_sam2.visualization_utils import (
    create_training_visualizer,
    visualize_from_probabilities,
)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HEALTHY_MEAN = (0.238, 0.238, 0.238)
HEALTHY_STD = (0.168, 0.168, 0.168)


NORMALIZATION_PRESETS: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "healthy": (HEALTHY_MEAN, HEALTHY_STD),
}


def resolve_normalization(name: Optional[str]) -> Tuple[bool, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Map a normalization preset name to mean/std tuples and whether normalization should be applied.
    """
    if name is None:
        return False, IMAGENET_MEAN, IMAGENET_STD
    if isinstance(name, str):
        key = name.strip().lower()
        if key in ("", "none", "null"):
            return False, IMAGENET_MEAN, IMAGENET_STD
        if key in NORMALIZATION_PRESETS:
            mean, std = NORMALIZATION_PRESETS[key]
            return True, mean, std
    raise ValueError(f"Unknown normalization preset '{name}'.")


def freeze_image_encoder_parameters(model):
    base_model = model.module if hasattr(model, 'module') else model
    frozen = 0
    for name, param in base_model.named_parameters():
        if name.startswith('image_encoder.'):
            if param.requires_grad:
                frozen += param.numel()
            param.requires_grad = False
    return frozen


def iter_with_progress(loader, desc):
    try:
        total = len(loader)
    except TypeError:
        total = None
    return tqdm(loader, desc=desc, total=total, dynamic_ncols=True)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class PairedImageMaskDataset(Dataset):
    """Dataset that pairs images and masks located in separate directories."""

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_size: int,
        num_classes: Optional[int] = None,
        ignore_index: Optional[int] = None,
        augment: bool = False,
        pad_only: bool = False,
        pad_center: bool = False,
        norm_mean: Tuple = IMAGENET_MEAN,
        norm_std: Tuple = IMAGENET_STD,
        apply_norm: bool = True,
        rotate_angle: int = 5,
        bc_strength: float = 0.2,
        noise_std: float = 0.0,
        blur_p: float = 0.0,
        blur_k: int = 3,
        elastic_p: float = 0.0,
        grid_p: float = 0.0,
        gamma_min: float = 0.9,
        gamma_max: float = 1.1,
        use_clahe: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        mask_lookup = {}
        mask_candidates = [
            p
            for p in self.mask_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not mask_candidates:
            raise ValueError(f"No mask files found under {self.mask_dir}")
        for mask_path in mask_candidates:
            stem = mask_path.stem
            if stem in mask_lookup:
                logging.warning(
                    "Duplicate mask stem '%s' found; keeping first instance.", stem
                )
                continue
            mask_lookup[stem] = mask_path

        image_candidates = [
            p
            for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_candidates:
            raise ValueError(f"No image files found under {self.image_dir}")

        self.pairs = []
        for image_path in sorted(image_candidates):
            mask_path = mask_lookup.get(image_path.stem)
            if mask_path is None:
                logging.warning(
                    "Skipping '%s' because no matching mask was found.", image_path.name
                )
                continue
            self.pairs.append((image_path, mask_path))

        if not self.pairs:
            raise ValueError(
                "No image/mask pairs found. Ensure filenames match between directories."
            )

        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.apply_norm = apply_norm
        self.image_size = image_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.augment = augment
        self.rotate_angle = max(0, int(rotate_angle))
        self.bc_strength = max(0.0, float(bc_strength))
        self.noise_std = max(0.0, float(noise_std))
        self.blur_p = float(max(0.0, min(1.0, blur_p)))
        blur_kernel = int(blur_k)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        if blur_kernel < 3:
            blur_kernel = 3
        self.blur_kernel = blur_kernel
        self.elastic_p = float(max(0.0, min(1.0, elastic_p)))
        self.grid_p = float(max(0.0, min(1.0, grid_p)))
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        if self.gamma_min <= 0 or self.gamma_max <= 0 or self.gamma_min > self.gamma_max:
            raise ValueError("gamma_min and gamma_max must be positive with gamma_min <= gamma_max.")
        self.use_clahe = bool(use_clahe)
        self.pad_only = pad_only
        if self.pad_only and self.ignore_index is None:
            raise ValueError("pad_only=True requires a valid ignore_index to mask padded pixels.")
        self.pad_center = pad_center
        if self.pad_center and not self.pad_only:
            raise ValueError("pad_center=True requires pad_only=True.")
        self.augmentations = self._build_augmentations() if augment else None
        self.use_cropping = False
        self.crop_frequency = 0.0
        self.crop_target_classes: list[int] = []
        self.crop_background_class = 0
        self.crop_min_margin = 123
        self.crop_min_crop = 256
        self.crop_extra_margin = 150
        self.crop_extra_down = 50
        self.crop_max_attempts = 50
        resampling = getattr(Image, 'Resampling', None)
        if resampling is not None:
            self._image_resample = resampling.BILINEAR
            self._mask_resample = resampling.NEAREST
        else:
            self._image_resample = Image.BILINEAR
            self._mask_resample = Image.NEAREST

    def __len__(self) -> int:
        return len(self.pairs)

    def _build_augmentations(self):
        transforms = [A.HorizontalFlip(p=0.8)]

        if self.rotate_angle > 0:
            transforms.append(
                A.Rotate(
                    limit=[-self.rotate_angle, self.rotate_angle],
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    rotate_method="ellipse",
                    crop_border=False,
                    mask_interpolation=cv2.INTER_NEAREST,
                    fill=0,
                    fill_mask=0,
                    p=0.8,
                )
            )

        if self.blur_p > 0.0:
            transforms.append(
                A.GaussianBlur(
                    blur_limit=self.blur_kernel,
                    sigma_limit=[0.5, 3],
                    p=self.blur_p,
                )
            )

        if self.elastic_p > 0.0:
            transforms.append(
                A.ElasticTransform(
                    alpha=1.0,
                    sigma=50.0,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=self.elastic_p,
                )
            )

        if self.grid_p > 0.0:
            transforms.append(
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=self.grid_p,
                )
            )

        if self.noise_std > 0.0:
            transforms.append(
                A.GaussNoise(
                    std_range=[0.0, self.noise_std],
                    mean_range=[0,0],
                    per_channel=False,
                    p=0.5,
                )
            )

        if self.use_clahe:
            transforms.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5))

        if self.bc_strength > 0.0:
            lower = max(0.0, 1.0 - self.bc_strength)
            upper = 1.0 + self.bc_strength
            transforms.append(
                A.ColorJitter(
                    brightness=[lower, upper],
                    contrast=[lower, upper],
                    saturation=[lower, upper],
                    hue=[-self.bc_strength, self.bc_strength],
                    p=0.8,
                )
            )

        gamma_min_int = max(1, int(round(self.gamma_min * 100)))
        gamma_max_int = max(gamma_min_int, int(round(self.gamma_max * 100)))
        transforms.append(
            A.RandomGamma(
                gamma_limit=(gamma_min_int, gamma_max_int),
                p=0.5,
            )
        )

        try:
            return A.Compose(transforms)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning(
                "Failed to initialize Albumentations pipeline (%s); using simple flip augmentations instead.",
                exc,
            )
            return None

    def configure_cropping(
        self,
        enabled: bool,
        crop_frequency: float,
        target_classes: Optional[Sequence[int]] = None,
        background_class: int = 0,
        min_margin: int = 123,
        min_crop: int = 256,
        extra_margin: int = 150,
        extra_down: int = 50,
        max_attempts: int = 50,
    ) -> None:
        self.use_cropping = enabled and crop_frequency > 0.0
        self.crop_frequency = float(np.clip(crop_frequency, 0.0, 1.0))
        self.crop_background_class = background_class
        self.crop_min_margin = min_margin
        self.crop_min_crop = min_crop
        self.crop_extra_margin = extra_margin
        self.crop_extra_down = extra_down
        self.crop_max_attempts = max(1, max_attempts)
        if target_classes:
            self.crop_target_classes = [
                int(cls) for cls in target_classes if int(cls) != background_class
            ]
        elif self.num_classes is not None and self.num_classes > 2:
            middle_class = self.num_classes // 2
            if middle_class == background_class and middle_class + 1 < self.num_classes:
                middle_class += 1
            self.crop_target_classes = [middle_class]
        else:
            self.crop_target_classes = []
        if self.use_cropping and not self.crop_target_classes:
            logging.warning(
                "Cropping enabled but no valid target classes found; disabling cropping."
            )
            self.use_cropping = False

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image_np = np.array(image)
        mask_np = np.array(mask, dtype=np.uint8)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image_np, mask=mask_np)
            image_np = augmented["image"]
            mask_np = augmented["mask"]
            if mask_np.dtype != np.uint8:
                mask_np = mask_np.astype(np.uint8, copy=False)

        if self.use_cropping and random.random() < self.crop_frequency:
            cropped = self._maybe_apply_crop(image_np, mask_np)
            if cropped is not None:
                image_np, mask_np = cropped

        if self.pad_only:
            padded_image_np, mask_array = self._pad_sample_numpy(image_np, mask_np)
            image = Image.fromarray(padded_image_np)
        else:
            image = Image.fromarray(image_np)
            mask = Image.fromarray(mask_np)

            mask = mask.resize((self.image_size, self.image_size), resample=self._mask_resample)
            mask_array = np.array(mask, dtype=np.int64)
            if mask_array.ndim == 3:
                mask_array = mask_array[..., 0]

            image = image.resize((self.image_size, self.image_size), resample=self._image_resample)

        mask_tensor = torch.tensor(mask_array, dtype=torch.long)
        if self.num_classes is not None:
            invalid = mask_tensor >= self.num_classes
            if invalid.any():
                if self.ignore_index is not None:
                    mask_tensor[invalid] = self.ignore_index
                else:
                    mask_tensor[invalid] = self.num_classes - 1
        if self.ignore_index is not None:
            negative = mask_tensor < 0
            if negative.any():
                mask_tensor[negative] = self.ignore_index

        image_tensor_unNorm = TF.to_tensor(image)
        if self.apply_norm:
            image_tensor = TF.normalize(image_tensor_unNorm, self.norm_mean, self.norm_std)
        else:
            image_tensor = image_tensor_unNorm

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "unNorm_image": image_tensor_unNorm
        }

    def _pad_sample_numpy(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Zero-pad image and mask to the configured image_size."""
        h, w = image_np.shape[:2]
        target = self.image_size
        if h > target or w > target:
            raise ValueError(
                f"Pad-only mode requires image dimensions <= image_size ({target}); "
                f"received image size ({h}, {w})."
            )

        pad_top = (target - h) // 2 if self.pad_center else 0
        pad_left = (target - w) // 2 if self.pad_center else 0
        pad_bottom = pad_top + h
        pad_right = pad_left + w
        padded_image = np.zeros((target, target, image_np.shape[2]), dtype=image_np.dtype)
        padded_image[pad_top:pad_bottom, pad_left:pad_right] = image_np

        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        mask_np = mask_np.astype(np.int64, copy=False)
        pad_fill = self.ignore_index if self.ignore_index is not None else 0
        padded_mask = np.full((target, target), pad_fill, dtype=np.int64)
        mask_h, mask_w = mask_np.shape
        mask_top = (target - mask_h) // 2 if self.pad_center else 0
        mask_left = (target - mask_w) // 2 if self.pad_center else 0
        mask_bottom = mask_top + mask_h
        mask_right = mask_left + mask_w
        padded_mask[mask_top:mask_bottom, mask_left:mask_right] = mask_np
        return padded_image, padded_mask

    def _maybe_apply_crop(
        self, image_np: np.ndarray, mask_np: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.crop_target_classes:
            return None

        candidate_classes = self.crop_target_classes.copy()
        random.shuffle(candidate_classes)
        for target_class in candidate_classes:
            try:
                bounds = self._sample_padded_crop(mask_np, target_class)
            except (RuntimeError, ValueError):
                continue
            if bounds is None:
                continue
            top, bottom, left, right = bounds
            cropped_img = image_np[top:bottom, left:right]
            cropped_mask = mask_np[top:bottom, left:right]
            if cropped_img.size == 0 or cropped_mask.size == 0:
                continue
            return cropped_img, cropped_mask
        return None

    def _sample_padded_crop(
        self, mask: np.ndarray, target_class: int
    ) -> Optional[Tuple[int, int, int, int]]:
        h, w = mask.shape[:2]
        coords = np.argwhere(mask == target_class)
        if coords.size == 0:
            raise ValueError(f"No pixels found for target class {target_class}.")

        min_margin = self.crop_min_margin
        valid = [
            (y, x)
            for y, x in coords
            if (min_margin <= y < h - min_margin) and (min_margin <= x < w - min_margin)
        ]
        if not valid:
            raise ValueError(
                f"No candidate pixels satisfy the margin constraint for class {target_class}."
            )

        background_class = self.crop_background_class
        for _ in range(self.crop_max_attempts):
            center_y, center_x = random.choice(valid)

            up = center_y
            while up >= 0 and mask[up, center_x] != background_class:
                up -= 1

            down = center_y
            while down < h and mask[down, center_x] != background_class:
                down += 1

            if up < 0 or down >= h:
                continue

            band_height = down - up - 1
            crop_size = max(band_height + self.crop_extra_margin + self.crop_extra_down, self.crop_min_crop)
            crop_size = min(crop_size, min(h, w))
            if crop_size < self.crop_min_crop:
                continue

            band_top = up + 1
            band_bottom = down - 1
            top_low = max(band_bottom + 1 - crop_size, 0)
            top_high = min(band_top, h - crop_size)
            if top_low > top_high:
                continue

            top = int(np.clip(center_y - crop_size // 2 + self.crop_extra_down, top_low, top_high))
            bottom = top + crop_size
            if bottom > h:
                shift = bottom - h
                top -= shift
                bottom -= shift
            top = max(top, 0)
            bottom = min(bottom, h)

            horizontal_size = crop_size
            left = int(np.clip(center_x - horizontal_size // 2, 0, w - horizontal_size))
            right = left + horizontal_size
            if right > w:
                shift = right - w
                left -= shift
                right -= shift
            left = max(left, 0)
            right = min(right, w)

            if bottom - top <= 0 or right - left <= 0:
                continue

            return top, bottom, left, right

        raise RuntimeError("Failed to sample a valid crop after multiple attempts.")


class SegmentationMetrics:
    """Utility to accumulate segmentation metrics across batches."""

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.total_correct = 0.0
        self.total_pixels = 0.0
        self.intersections = np.zeros(self.num_classes, dtype=np.float64)
        self.unions = np.zeros(self.num_classes, dtype=np.float64)
        self.pred_pixels = np.zeros(self.num_classes, dtype=np.float64)
        self.target_pixels = np.zeros(self.num_classes, dtype=np.float64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        logits = logits.detach()
        preds = logits.argmax(dim=1).cpu()
        targets = targets.detach().cpu()
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
        else:
            valid = torch.ones_like(targets, dtype=torch.bool)
        valid_pixels = valid.sum().item()
        if valid_pixels == 0:
            return
        self.total_pixels += valid_pixels
        self.total_correct += ((preds == targets) & valid).sum().item()
        for class_idx in range(self.num_classes):
            pred_c = (preds == class_idx) & valid
            target_c = (targets == class_idx) & valid
            intersection = torch.logical_and(pred_c, target_c).sum().item()
            pred_sum = pred_c.sum().item()
            target_sum = target_c.sum().item()
            union = pred_sum + target_sum - intersection
            self.intersections[class_idx] += intersection
            self.unions[class_idx] += union
            self.pred_pixels[class_idx] += pred_sum
            self.target_pixels[class_idx] += target_sum

    def compute(self) -> Dict[str, float]:
        pixel_acc = (
            self.total_correct / self.total_pixels if self.total_pixels > 0 else 0.0
        )
        per_class_iou = {}
        per_class_dice = {}
        iou_values = []
        dice_values = []
        iou_values_excl_bg = []
        dice_values_excl_bg = []
        for class_idx in range(self.num_classes):
            intersection = self.intersections[class_idx]
            union = self.unions[class_idx]
            pred_sum = self.pred_pixels[class_idx]
            target_sum = self.target_pixels[class_idx]

            if union == 0 and pred_sum == 0 and target_sum == 0:
                iou = 1.0
            elif union > 0:
                iou = intersection / union
            else:
                iou = 0.0

            denom = pred_sum + target_sum
            if denom == 0:
                dice = 1.0
            else:
                dice = (2.0 * intersection) / denom

            per_class_iou[f"class_{class_idx}_iou"] = float(iou)
            per_class_dice[f"class_{class_idx}_dice"] = float(dice)
            iou_values.append(iou)
            dice_values.append(dice)
            if class_idx != 0:
                iou_values_excl_bg.append(iou)
                dice_values_excl_bg.append(dice)

        mean_iou = float(np.mean(iou_values)) if iou_values else 0.0
        mean_dice = float(np.mean(dice_values)) if dice_values else 0.0
        mean_iou_excl_bg = float(np.mean(iou_values_excl_bg)) if iou_values_excl_bg else mean_iou
        mean_dice_excl_bg = float(np.mean(dice_values_excl_bg)) if dice_values_excl_bg else mean_dice

        metrics = {
            "pixel_acc": float(pixel_acc),
            "iou_score": mean_iou,
            "iou_score_exclB": mean_iou_excl_bg,
            "dice_score": mean_dice,
            "dice_score_exclB": mean_dice_excl_bg,
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
        }
        metrics.update(per_class_iou)
        metrics.update(per_class_dice)
        return metrics








class GradientMonitor:
    """Track gradient norms for selected parameter groups each training step."""

    def __init__(self, model: torch.nn.Module) -> None:
        base_model = model.module if hasattr(model, "module") else model
        param_dict = dict(base_model.named_parameters())

        self.groups: Dict[str, list[torch.nn.Parameter]] = {}

        mask_names = [
            "sam_mask_decoder.mask_tokens.weight",
            "sam_mask_decoder.iou_token.weight",
        ]
        if getattr(base_model.sam_mask_decoder, "pred_obj_scores", False):
            mask_names.append("sam_mask_decoder.obj_score_token.weight")
        mask_params = [param_dict[name] for name in mask_names if name in param_dict]
        if mask_params:
            self.groups["mask_tokens"] = mask_params

        backbone_params = [
            param
            for name, param in param_dict.items()
            if name.startswith("image_encoder.") and param.requires_grad
        ]
        if backbone_params:
            self.groups["image_encoder"] = backbone_params

        decoder_params = [
            param
            for name, param in param_dict.items()
            if name.startswith("sam_mask_decoder.") and param.requires_grad
        ]
        if decoder_params:
            self.groups.setdefault("sam_mask_decoder", decoder_params)

        self._step_stats: Dict[str, list[tuple[float, float]]] = defaultdict(list)

    def start_epoch(self) -> None:
        self._step_stats = defaultdict(list)

    def record_step(self) -> None:
        if not self.groups:
            return
        for group_name, params in self.groups.items():
            norms = [
                param.grad.detach().norm()
                for param in params
                if param.grad is not None
            ]
            if not norms:
                continue
            stacked = torch.stack(norms)
            mean_val = stacked.mean().item()
            max_val = stacked.max().item()
            self._step_stats[group_name].append((mean_val, max_val))

    def end_epoch(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for group_name, stats in self._step_stats.items():
            if not stats:
                continue
            stats_arr = np.array(stats, dtype=np.float64)
            mean_series = stats_arr[:, 0]
            max_series = stats_arr[:, 1]
            metrics[f"grad_norm/{group_name}_mean"] = float(mean_series.mean())
            metrics[f"grad_norm/{group_name}_max"] = float(max_series.max())
            metrics[f"grad_norm/{group_name}_last"] = float(mean_series[-1])
        self._step_stats = defaultdict(list)
        return metrics

def forward_semantic_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run the Semantic SAM2 model and return per-class logits."""
    backbone_out = model.forward_image(images)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    batch_size = images.size(0)
    if len(vision_feats) > 1:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
        high_res_features = [
            feat.permute(1, 2, 0).reshape(batch_size, feat.size(2), *size)
            for feat, size in zip(vision_feats[:-1], feat_sizes[:-1])
        ]
    else:
        high_res_features = None
    fused_features = vision_feats[-1].permute(1, 2, 0).reshape(
        batch_size, vision_feats[-1].size(2), *feat_sizes[-1]
    )
    sam_outputs = model._forward_sam_heads(
        backbone_features=fused_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=high_res_features,
        multimask_output=False,
    )
    _, high_res_multimasks, _, _, _, _, _ = sam_outputs
    return high_res_multimasks


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[amp.GradScaler],
    grad_clip: float,
    num_classes: int,
    ignore_index: Optional[int],
    grad_monitor: Optional[GradientMonitor] = None,
    progress_desc: str = "Train",
) -> Dict[str, float]:
    model.train()
    metric_tracker = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    running_loss = 0.0
    num_samples = 0
    use_amp = scaler is not None and scaler.is_enabled()
    if grad_monitor is not None:
        grad_monitor.start_epoch()

    for batch in iter_with_progress(dataloader, progress_desc):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with amp.autocast():
                logits = forward_semantic_logits(model, images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = forward_semantic_logits(model, images)
            loss = criterion(logits, targets)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if grad_monitor is not None:
            grad_monitor.record_step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size
        metric_tracker.update(logits, targets)

    avg_loss = running_loss / max(num_samples, 1)
    metrics = metric_tracker.compute()
    metrics["total_loss"] = float(running_loss)
    metrics["avg_loss"] = float(avg_loss)
    metrics["loss"] = float(avg_loss)
    if grad_monitor is not None:
        metrics.update(grad_monitor.end_epoch())
    return metrics


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: Optional[int],
    progress_desc: str = "Val",
) -> Dict[str, float]:
    model.eval()
    metric_tracker = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    running_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in iter_with_progress(dataloader, progress_desc):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            logits = forward_semantic_logits(model, images)
            loss = criterion(logits, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size
            metric_tracker.update(logits, targets)

    avg_loss = running_loss / max(num_samples, 1)
    metrics = metric_tracker.compute()
    metrics["total_loss"] = float(running_loss)
    metrics["avg_loss"] = float(avg_loss)
    metrics["loss"] = float(avg_loss)
    model.train()
    return metrics


def flatten_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    """Prefix metric keys to make them W&B friendly."""
    flat = {}
    for key, value in metrics.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        flat[f"{prefix}/{key}"] = numeric
    return flat


@torch.no_grad()
def visualize_predictions(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    visualizer,
    wandb_wrapper,
    epoch: int,
    output_dir: Path,
    num_samples: int = 3,
) -> None:
    """Render a few validation predictions and log/save the figures."""
    if visualizer is None or len(dataset) == 0 or num_samples <= 0:
        return

    viz_dir = output_dir / f"visualizations_epoch_{epoch:04d}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    model_was_training = model.training
    model.eval()

    try:
        total = min(num_samples, len(dataset))
        for sample_idx in range(total):
            sample = dataset[sample_idx]
            image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
            logits = forward_semantic_logits(model, image)
            probs = torch.softmax(logits, dim=1)[0].cpu()
            target_mask = sample["mask"].cpu()

            fig, iou_scores = visualize_from_probabilities(
                pred_probs=probs,
                target_mask=target_mask,
                visualizer=visualizer,
                title=f"Validation Sample {sample_idx}",
                epoch=epoch,
                use_argmax=True,
            )

            source_name = Path(sample.get("image_path", f"val_{sample_idx}")).stem
            if wandb_wrapper is not None:
                wandb_wrapper.log_figure_with_metrics(
                    prefix="val_visualization",
                    source_idx=sample_idx,
                    source_name=source_name,
                    epoch=epoch,
                    fig=fig,
                    iou_scores=iou_scores,
                    close_figure=False,
                )

            out_path = viz_dir / f"{source_name}_epoch{epoch:04d}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
    finally:
        if model_was_training:
            model.train()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
    scaler: Optional[amp.GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    best_metric: Optional[float] = None,
) -> None:
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if scaler is not None and scaler.is_enabled():
        state["scaler_state"] = scaler.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if best_metric is not None:
        state["best_metric"] = best_metric
    torch.save(state, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Semantic SAM2 on paired image/mask datasets.")
    parser.add_argument("--train-images", type=Path, default=Path(r"D:\GitHub\segment_anything_private\processed_data\Healthy\train\img_unchanged"), help="Directory with training images.")
    parser.add_argument("--train-masks", type=Path, default=Path(r"D:\GitHub\segment_anything_private\processed_data\Healthy\train\mask_unchanged"), help="Directory with training masks.")
    parser.add_argument("--val-images", type=Path, default=Path(r"D:\GitHub\segment_anything_private\processed_data\Healthy\val\img_unchanged"), help="Directory with validation images.")
    parser.add_argument("--val-masks", type=Path, default=Path(r"D:\GitHub\segment_anything_private\processed_data\Healthy\val\mask_unchanged"), help="Directory with validation masks.")
    parser.add_argument("--config-file", type=str, default="configs/sam2.1/sam2.1_hiera_b+.yaml", help="Hydra config name for the backbone (e.g. configs/sam2.1/sam2.1_hiera_b+.yaml).")
    # parser.add_argument("--checkpoint", type=str, default=root_dir/"checkpoints/sam2.1_hiera_base_plus.pt", help="Optional checkpoint path to initialize the model.")
    parser.add_argument("--checkpoint", type=str, default=root_dir/"semantic_sam2/training_checkpoints/FIE_best.pt", help="Optional checkpoint path to initialize the model.")
    parser.add_argument("--num-classes", type=int, default=9, help="Number of semantic classes (including background).")
    parser.add_argument("--use-safe-checkpoint", default=True, action="store_true", help="Use staged checkpoint loading to adapt mismatched decoders.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use (e.g. cuda, cuda:0, cpu).")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for decoder and non-encoder parameters.")
    parser.add_argument("--lr-image-encoder", type=float, default=5e-5, help="Learning rate for image encoder parameters (if trainable).")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader worker threads.")
    parser.add_argument("--amp", default=False, action="store_true", help="Enable mixed-precision training if CUDA is available.")
    parser.add_argument("--freeze-image-encoder", default=True, action="store_true", help="Freeze all parameters in the image encoder during fine-tuning.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clip value (0 disables clipping).")
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="cosine", help="Learning rate schedule.")
    parser.add_argument("--output-dir", type=Path, default=Path("semantic_sam2_runs"), help="Directory to store checkpoints.")
    parser.add_argument("--ignore-index", type=int, default=-1, help="Ignore index value used in the masks.")
    parser.add_argument("--augment", default=True, action="store_true", help="Enable simple flip augmentations.")
    parser.add_argument("--aug-rot-angle", type=int, default=5, help="Maximum absolute rotation angle (degrees) applied when augmentation is enabled.")
    parser.add_argument("--img-normalization", type=str, choices=["none", "imagenet", "healthy"], default="imagenet", help="Normalization preset applied to input images ('none' disables normalization).")
    parser.add_argument("--bc-strength", type=float, default=0.2, help="Brightness/contrast jitter strength (0 disables).")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Standard deviation for Gaussian noise as a fraction of 255 (0 disables).")
    parser.add_argument("--blur-p", type=float, default=0.0, help="Probability of applying Gaussian blur (0 disables).")
    parser.add_argument("--blur-k", type=int, default=3, help="Kernel size for Gaussian blur (odd integer >=3).")
    parser.add_argument("--elastic-p", type=float, default=0.0, help="Probability of applying elastic deformation (0 disables).")
    parser.add_argument("--grid-p", type=float, default=0.0, help="Probability of applying grid distortion (0 disables).")
    parser.add_argument("--gamma-min", type=float, default=0.9, help="Lower bound for random gamma adjustment.")
    parser.add_argument("--gamma-max", type=float, default=1.1, help="Upper bound for random gamma adjustment.")
    parser.add_argument("--use-clahe", type=int, choices=[0, 1], default=0, help="Apply CLAHE (1) or skip (0).")
    parser.add_argument("--pad-only", default=True, action="store_true", help="Zero-pad inputs to image_size instead of resizing.")
    parser.add_argument("--center-pad", default=True, action="store_true", help="Center content when using pad-only mode.")
    parser.add_argument("--focal-gamma", type=float, default=3.0, help="Gamma parameter for the focal loss component.")
    parser.add_argument("--ce-class-weights", type=str, default="1,1,1,1.3,1,1,1,1.3,1", help="Comma-separated list of class weights applied to the cross-entropy term.")
    parser.add_argument("--enable-cropping", default=False, action="store_true", help="Enable class-focused cropping augmentation.")
    parser.add_argument("--crop-frequency", type=float, default=0.0, help="Probability of applying a focused crop to a sample when cropping is enabled.")
    parser.add_argument("--crop-target-classes", type=str, default=None, help="Comma-separated list of target classes to bias cropping towards (defaults to middle class).")
    parser.add_argument("--crop-min-margin", type=int, default=123, help="Minimum distance from image border for crop seed points.")
    parser.add_argument("--crop-min-size", type=int, default=256, help="Minimum crop edge length before resizing to the model input size.")
    parser.add_argument("--crop-extra-margin", type=int, default=150, help="Additional padding added to the class band when computing crop size.")
    parser.add_argument("--crop-extra-down", type=int, default=50, help="Extra downward padding to include below the sampled center point.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--deterministic", default=True, action="store_true", help="Enable deterministic training (may reduce throughput).")
    parser.add_argument("--wandb-project", type=str, default="sam2_NR206", help="Weights & Biases project name.")
    parser.add_argument("--wandb-run-name", type=str, default="local_test", help="Optional W&B run name override.")
    parser.add_argument("--disable-wandb", default=True, action="store_true", help="Disable W&B logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_seed(args.seed, deterministic=args.deterministic)
    if args.center_pad and not args.pad_only:
        raise SystemExit("--center-pad requires --pad-only to be set.")

    try:
        apply_norm, norm_mean, norm_std = resolve_normalization(args.img_normalization)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.gamma_min <= 0 or args.gamma_max <= 0:
        raise SystemExit("--gamma-min and --gamma-max must be positive.")
    if args.gamma_min > args.gamma_max:
        raise SystemExit("--gamma-min must be <= --gamma-max.")
    use_clahe = bool(args.use_clahe)

    ce_class_weights: Optional[torch.Tensor] = None
    if args.ce_class_weights:
        try:
            weights = [float(part) for part in args.ce_class_weights.split(",")]
        except ValueError as exc:
            raise SystemExit(f"Failed to parse --ce-class-weights '{args.ce_class_weights}': {exc}") from exc
        if len(weights) != args.num_classes:
            raise SystemExit(
                f"Expected {args.num_classes} weights in --ce-class-weights but received {len(weights)}."
            )
        ce_class_weights = torch.tensor(weights, dtype=torch.float32)


    crop_target_classes: Optional[list[int]] = None
    if args.crop_target_classes:
        try:
            crop_target_classes = [int(part) for part in args.crop_target_classes.split(",")]
        except ValueError as exc:
            raise SystemExit(
                f"Failed to parse --crop-target-classes '{args.crop_target_classes}': {exc}"
            ) from exc
        invalid_classes = [
            cls for cls in crop_target_classes if cls < 0 or cls >= args.num_classes
        ]
        if invalid_classes:
            raise SystemExit(
                f"Crop target classes {invalid_classes} fall outside the valid range [0, {args.num_classes - 1}]."
            )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but unavailable; falling back to CPU.")
            device = torch.device("cpu")
    logging.info("Using device: %s", device)

    hydra_overrides = [
        "++model._target_=semantic_sam2.semantic_sam2_components.SAM2Semantic",
        f"++model.num_classes={args.num_classes}",
        "++model.num_maskmem=0",
        "++model.use_mask_input_as_output_without_sam=false",
        "++model.multimask_output_in_sam=false",
        "++model.pred_obj_scores=false",
        "++model.pred_obj_scores_mlp=false",
        "++model.fixed_no_obj_ptr=false",
        # "++model.sam_mask_decoder_extra_args.hypernet_output_dim=64",
    ]

    logging.info("Building Semantic SAM2 model from config '%s'", args.config_file)
    model = build_semantic_sam2(
        config_file=args.config_file,
        ckpt_path=args.checkpoint,
        device=str(device),
        mode="train",
        hydra_overrides_extra=hydra_overrides,
        apply_postprocessing=False,
        use_load_checkpoint_staged_safe=args.use_safe_checkpoint,
    )
    model.num_maskmem = 0
    model.use_mask_input_as_output_without_sam = False

    frozen_image_params = 0
    if args.freeze_image_encoder:
        logging.info('Freezing image encoder parameters')
        frozen_image_params = freeze_image_encoder_parameters(model)
        if frozen_image_params == 0:
            logging.warning("No parameters matched the 'image_encoder.' prefix; encoder might already be frozen or unavailable.")
        else:
            logging.info('Froze %d parameters within the image encoder', frozen_image_params)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('Model parameters: %d trainable out of %d total', trainable_params, total_params)

    train_dataset = PairedImageMaskDataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        image_size=model.image_size,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        augment=args.augment,
        norm_mean=norm_mean,
        norm_std=norm_std,
        apply_norm=apply_norm,
        rotate_angle=args.aug_rot_angle,
        bc_strength=args.bc_strength,
        noise_std=args.noise_std,
        blur_p=args.blur_p,
        blur_k=args.blur_k,
        elastic_p=args.elastic_p,
        grid_p=args.grid_p,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        use_clahe=use_clahe,
        pad_only=args.pad_only,
        pad_center=args.center_pad,
    )
    # train_dataset.configure_cropping(
    #     enabled=args.enable_cropping,
    #     crop_frequency=args.crop_frequency,
    #     target_classes=crop_target_classes,
    #     background_class=0,
    #     min_margin=args.crop_min_margin,
    #     min_crop=args.crop_min_size,
    #     extra_margin=args.crop_extra_margin,
    #     extra_down=args.crop_extra_down,
    # )
    train_dataset.configure_cropping(
        enabled=False,
        crop_frequency=0.0,
    )
    val_dataset = PairedImageMaskDataset(
        image_dir=args.val_images,
        mask_dir=args.val_masks,
        image_size=model.image_size,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        augment=False,
        norm_mean=norm_mean,
        norm_std=norm_std,
        apply_norm=apply_norm,
        rotate_angle=args.aug_rot_angle,
        bc_strength=args.bc_strength,
        noise_std=args.noise_std,
        blur_p=args.blur_p,
        blur_k=args.blur_k,
        elastic_p=args.elastic_p,
        grid_p=args.grid_p,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        use_clahe=use_clahe,
        pad_only=args.pad_only,
        pad_center=args.center_pad,
    )
    val_dataset.configure_cropping(
        enabled=False,
        crop_frequency=0.0,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=pin_memory,
        drop_last=False,
    )

    criterion = FocalLossCEAligned(
        gamma=args.focal_gamma,
        weight=ce_class_weights,
        ignore_index=args.ignore_index,
        reduction='mean'
    )
    base_model = model.module if hasattr(model, "module") else model
    encoder_params = []
    non_encoder_params = []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("image_encoder."):
            encoder_params.append(param)
        else:
            non_encoder_params.append(param)

    if not encoder_params and not non_encoder_params:
        raise RuntimeError("No trainable parameters found for optimizer initialization.")

    optimizer_param_groups = []
    if non_encoder_params:
        optimizer_param_groups.append({"params": non_encoder_params, "lr": args.lr})
    if encoder_params:
        optimizer_param_groups.append({"params": encoder_params, "lr": args.lr_image_encoder})

    optimizer = torch.optim.AdamW(optimizer_param_groups, weight_decay=args.weight_decay)

    def _count_params(param_list: list[torch.nn.Parameter]) -> int:
        return sum(p.numel() for p in param_list)

    logging.info(
        "Optimizer groups: decoder/non-encoder=%d params @ lr=%.2e, encoder=%d params @ lr=%.2e",
        _count_params(non_encoder_params),
        args.lr,
        _count_params(encoder_params),
        args.lr_image_encoder if encoder_params else 0.0,
    )

    grad_monitor = GradientMonitor(model)

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    amp_enabled = args.amp and device.type == "cuda"
    scaler = amp.GradScaler(enabled=amp_enabled)

    wandb_wrapper = create_wandb_wrapper(enabled=not args.disable_wandb)
    if wandb_wrapper.enabled:
        wandb_config = {
            "config_file": args.config_file,
            "checkpoint": args.checkpoint,
            "num_classes": args.num_classes,
            "lr": args.lr,
            "lr_image_encoder": args.lr_image_encoder,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "augment": args.augment,
            "pad_only": args.pad_only,
            "center_pad": args.center_pad,
            "freeze_image_encoder": args.freeze_image_encoder,
            "train_images": str(args.train_images),
            "train_masks": str(args.train_masks),
            "val_images": str(args.val_images),
            "val_masks": str(args.val_masks),
            "focal_gamma": args.focal_gamma,
            "ce_class_weights": args.ce_class_weights,
            "enable_cropping": args.enable_cropping,
            "crop_frequency": args.crop_frequency,
            "crop_target_classes": args.crop_target_classes,
            "crop_min_margin": args.crop_min_margin,
            "crop_min_size": args.crop_min_size,
            "crop_extra_margin": args.crop_extra_margin,
            "crop_extra_down": args.crop_extra_down,
        }
        wandb_kwargs = {}
        if args.wandb_run_name:
            wandb_kwargs["name"] = args.wandb_run_name
        wandb_wrapper.init(project=args.wandb_project, config=wandb_config, **wandb_kwargs)

    visualizer = create_training_visualizer(device=str(device), include_background_as_class=True)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    best_metric = float("-inf")
    best_epoch = 0

    try:
        for epoch in range(1, args.epochs + 1):
            logging.info("Epoch %d/%d", epoch, args.epochs)
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                grad_clip=args.grad_clip,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                grad_monitor=grad_monitor,
                progress_desc=f"Train {epoch}/{args.epochs}",
            )
            logging.info(
                "Train | loss: %.4f | acc: %.4f | mIoU: %.4f",
                train_metrics.get("loss", 0.0),
                train_metrics.get("pixel_acc", 0.0),
                train_metrics.get("mean_iou", 0.0),
            )

            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                progress_desc=f"Val {epoch}/{args.epochs}",
            )
            logging.info(
                "Val   | loss: %.4f | acc: %.4f | mIoU: %.4f",
                val_metrics.get("loss", 0.0),
                val_metrics.get("pixel_acc", 0.0),
                val_metrics.get("mean_iou", 0.0),
            )

            if scheduler is not None:
                scheduler.step()

            log_payload = flatten_metrics("train", train_metrics)
            log_payload.update(flatten_metrics("val", val_metrics))
            for split_name, split_metrics in (("train", train_metrics), ("val", val_metrics)):
                for metric_name, metric_value in split_metrics.items():
                    try:
                        numeric_value = float(metric_value)
                    except (TypeError, ValueError):
                        continue
                    log_payload[f"{split_name}_{metric_name}"] = numeric_value
            log_payload["epoch"] = float(epoch)
            log_payload["lr"] = float(optimizer.param_groups[0]["lr"])
            if wandb_wrapper.enabled:
                wandb_wrapper.log(log_payload, step=epoch)

            visualize_predictions(
                model=model,
                dataset=val_dataset,
                device=device,
                visualizer=visualizer,
                wandb_wrapper=wandb_wrapper,
                epoch=epoch,
                output_dir=output_dir,
            )

            current_metric = val_metrics.get("mean_iou", 0.0)
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
                best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=output_dir / "last.pt",
                scaler=scaler,
                scheduler=scheduler,
                best_metric=best_metric,
            )
            if is_best:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    path=output_dir / "best.pt",
                    scaler=scaler,
                    scheduler=scheduler,
                    best_metric=best_metric,
                )
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    finally:
        if wandb_wrapper.enabled:
            wandb_wrapper.finish()

    logging.info(
        "Best validation mIoU: %.4f at epoch %d", best_metric if best_metric > float("-inf") else 0.0, best_epoch
    )
    logging.info("Checkpoints saved to %s", output_dir.resolve())


if __name__ == "__main__":
    main()

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
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

from setup_imports import root_dir
from semantic_sam2.build_semantic_sam2 import build_semantic_sam2
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

        self.image_size = image_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.augment = augment
        resampling = getattr(Image, 'Resampling', None)
        if resampling is not None:
            self._image_resample = resampling.BILINEAR
            self._mask_resample = resampling.NEAREST
        else:
            self._image_resample = Image.BILINEAR
            self._mask_resample = Image.NEAREST

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        mask = mask.resize((self.image_size, self.image_size), resample=self._mask_resample)
        mask_array = np.array(mask, dtype=np.int64)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]

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

        image = image.resize((self.image_size, self.image_size), resample=self._image_resample)
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


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


def forward_semantic_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run the Semantic SAM2 model and return per-class logits."""
    backbone_out = model.forward_image(images)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    batch_size = images.size(0)
    if len(vision_feats) > 1:
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
    progress_desc: str = "Train",
) -> Dict[str, float]:
    model.train()
    metric_tracker = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    running_loss = 0.0
    num_samples = 0
    use_amp = scaler is not None and scaler.is_enabled()

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

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size
        metric_tracker.update(logits, targets)

    avg_loss = running_loss / max(num_samples, 1)
    metrics = metric_tracker.compute()
    metrics["total_loss"] = float(running_loss)
    metrics["avg_loss"] = float(avg_loss)
    metrics["loss"] = float(avg_loss)
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
    parser.add_argument("--train-images", type=Path, default=Path(r"D:\GitHub\segment_anything_private\processed_data\Healthy\train\img_stretched"), help="Directory with training images.")
    parser.add_argument("--train-masks", type=Path, default=Path(r"D:\GitHub\segment_anything_private\processed_data\Healthy\train\mask_stretched"), help="Directory with training masks.")
    parser.add_argument("--val-images", type=Path, default=Path(r"D:\GitHub\segment_anything_private\processed_data\Healthy\val\img_stretched"), help="Directory with validation images.")
    parser.add_argument("--val-masks", type=Path, default=Path(r"D:\GitHub\segment_anything_private\processed_data\Healthy\val\mask_stretched"), help="Directory with validation masks.")
    parser.add_argument("--config-file", type=str, default="configs/sam2.1/sam2.1_hiera_b+.yaml", help="Hydra config name for the backbone (e.g. configs/sam2.1/sam2.1_hiera_b+.yaml).")
    parser.add_argument("--checkpoint", type=str, default=root_dir/"checkpoints/sam2.1_hiera_base_plus.pt", help="Optional checkpoint path to initialize the model.")
    parser.add_argument("--num-classes", type=int, default=9, help="Number of semantic classes (including background).")
    parser.add_argument("--use-safe-checkpoint", default=True, action="store_true", help="Use staged checkpoint loading to adapt mismatched decoders.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use (e.g. cuda, cuda:0, cpu).")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per step.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader worker threads.")
    parser.add_argument("--amp", default=False, action="store_true", help="Enable mixed-precision training if CUDA is available.")
    parser.add_argument("--freeze-image-encoder", default=True, action="store_true", help="Freeze all parameters in the image encoder during fine-tuning.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clip value (0 disables clipping).")
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="none", help="Learning rate schedule.")
    parser.add_argument("--output-dir", type=Path, default=Path("semantic_sam2_runs"), help="Directory to store checkpoints.")
    parser.add_argument("--ignore-index", type=int, default=-1, help="Ignore index value used in the masks.")
    parser.add_argument("--augment", default=True, action="store_true", help="Enable simple flip augmentations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training (may reduce throughput).")
    parser.add_argument("--wandb-project", type=str, default="sam2_NR206", help="Weights & Biases project name.")
    parser.add_argument("--wandb-run-name", type=str, default="test_sam_2", help="Optional W&B run name override.")
    parser.add_argument("--disable-wandb", default=False, action="store_true", help="Disable W&B logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_seed(args.seed, deterministic=args.deterministic)

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
    )
    val_dataset = PairedImageMaskDataset(
        image_dir=args.val_images,
        mask_dir=args.val_masks,
        image_size=model.image_size,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        augment=False,
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

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

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
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "augment": args.augment,
            "freeze_image_encoder": args.freeze_image_encoder,
            "train_images": str(args.train_images),
            "train_masks": str(args.train_masks),
            "val_images": str(args.val_images),
            "val_masks": str(args.val_masks),
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

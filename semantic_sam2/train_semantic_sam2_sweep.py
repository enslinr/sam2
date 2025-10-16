#!/usr/bin/env python3
"""
Sweep-only training script for Semantic SAM2.

- No argparse; all knobs come from wandb.config (set by your sweep).
- Initialize W&B FIRST so sweep params apply before building model/datasets.
- Uses your existing training utilities (train_one_epoch, evaluate, etc.).
"""

import os
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.cuda import amp

import wandb

# --- Your project imports (assumes same package layout as your original script)
from setup_imports import root_dir
from semantic_sam2.build_semantic_sam2 import build_semantic_sam2
from semantic_sam2.train_semantic_sam2 import (
    set_seed,
    PairedImageMaskDataset,
    FocalLossCEAligned,
    freeze_image_encoder_parameters,
    GradientMonitor,
    train_one_epoch,
    evaluate,
    flatten_metrics,
    visualize_predictions,
    save_checkpoint,
    resolve_normalization,
)
from semantic_sam2.visualization_utils import create_training_visualizer

RUN_ON_CLUSTER = False


# --- Default configuration used when starting runs (sweeps will override)
if RUN_ON_CLUSTER:
    # Data
    train_images=str(Path("/home-mscluster/wroux/processed_data/Healthy/train/img_stretched"))
    train_masks=str(Path("/home-mscluster/wroux/processed_data/Healthy/train/mask_stretched"))
    val_images=str(Path("/home-mscluster/wroux/processed_data/Healthy/val/img_stretched"))
    val_masks=str(Path("/home-mscluster/wroux/processed_data/Healthy/val/mask_stretched"))
    freeze_image_encoder=False
else:
    # Data
    train_images=str(Path(r"D:\GitHub\segment_anything_private\processed_data\Healthy\train\img_unchanged"))
    train_masks=str(Path(r"D:\GitHub\segment_anything_private\processed_data\Healthy\train\mask_unchanged"))
    val_images=str(Path(r"D:\GitHub\segment_anything_private\processed_data\Healthy\val\img_unchanged"))
    val_masks=str(Path(r"D:\GitHub\segment_anything_private\processed_data\Healthy\val\mask_unchanged"))
    freeze_image_encoder=True




DEFAULTS = dict(
    # Data
    train_images=train_images,
    train_masks=train_masks,
    val_images=val_images,
    val_masks=val_masks,

    # Model/backbone
    config_file="configs/sam2.1/sam2.1_hiera_b+.yaml",
    checkpoint=str(root_dir / "checkpoints/sam2.1_hiera_base_plus.pt"),
    use_safe_checkpoint=True,
    num_classes=9,

    # Train loop
    device="cuda",
    epochs=200,
    # epochs=10,
    batch_size=2,
    num_workers=2,
    lr=2e-4,
    lr_image_encoder=5e-5,
    weight_decay=0.01,
    amp=True,
    grad_clip=1.0,
    lr_scheduler="cosine",
    freeze_image_encoder=freeze_image_encoder,

    # Loss
    focal_gamma=3.0,
    focal_weight=1.0,
    ce_class_weights="1,1,1,1.3,1,1,1,1.3,1",  # parsed to list later
    ignore_index=-1,

    # Augment & cropping
    augment=True,
    aug_rot_angle=5,
    img_normalization="imagenet",
    pad_only=True,
    bc_strength=0.2,
    noise_std=0.0,
    blur_p=0.0,
    blur_k=3,
    elastic_p=0.0,
    grid_p=0.0,
    gamma_min=0.9,
    gamma_max=1.1,
    use_clahe=0,
    enable_cropping=False,

    # Repro
    seed=42,
    deterministic=True,

    # Logging/output
    wandb_project="sam2_NR206_sweep",
    wandb_run_name=None,
    output_dir="semantic_sam2_runs/run_training_sweep",
)


def parse_ce_weights(s: Optional[str], num_classes: int) -> Optional[torch.Tensor]:
    if not s:
        return None
    try:
        weights = [float(part) for part in s.split(",")]
    except ValueError as exc:
        raise SystemExit(f"Failed to parse --ce-class-weights '{s}': {exc}") from exc
    if len(weights) != num_classes:
        raise SystemExit(
            f"Expected {num_classes} weights in --ce-class-weights but received {len(weights)}."
        )
    ce_class_weights = torch.tensor(weights, dtype=torch.float32)
    return ce_class_weights


def parse_crop_targets(s: Optional[str], num_classes: int) -> Optional[list[int]]:
    if not s:
        return None
    vals = [int(x) for x in s.split(",")]
    bad = [v for v in vals if v < 0 or v >= num_classes]
    if bad:
        raise ValueError(f"crop_target_classes out of range: {bad} for K={num_classes}")
    return vals


def train():
    # ---- 1) Init logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # ---- 2) Init W&B FIRST (sweep will inject params here)
    wandb.init(project=DEFAULTS["wandb_project"], config=DEFAULTS)
    cfg = wandb.config  # treat as dict-like

    # Optional explicit run name
    if cfg.get("wandb_run_name"):
        wandb.run.name = cfg["wandb_run_name"]

    # ---- 3) Seed & device
    set_seed(int(cfg["seed"]), deterministic=bool(cfg["deterministic"]))
    device = torch.device(str(cfg["device"]) if torch.cuda.is_available() or str(cfg["device"]) == "cpu" else "cpu")
    logging.info("Using device: %s", device)

    # ---- 4) Build model (after sweep params are set)
    hydra_overrides = [
        "++model._target_=semantic_sam2.semantic_sam2_components.SAM2Semantic",
        f"++model.num_classes={int(cfg['num_classes'])}",
        "++model.num_maskmem=0",
        "++model.use_mask_input_as_output_without_sam=false",
        "++model.multimask_output_in_sam=false",
        "++model.pred_obj_scores=false",
        "++model.pred_obj_scores_mlp=false",
        "++model.fixed_no_obj_ptr=false",
    ]
    model = build_semantic_sam2(
        config_file=str(cfg["config_file"]),
        ckpt_path=str(cfg["checkpoint"]),
        device=str(device),
        mode="train",
        hydra_overrides_extra=hydra_overrides,
        apply_postprocessing=False,
        use_load_checkpoint_staged_safe=bool(cfg["use_safe_checkpoint"]),
    )
    model.num_maskmem = 0
    model.use_mask_input_as_output_without_sam = False

    if bool(cfg["freeze_image_encoder"]):
        n_frozen = freeze_image_encoder_parameters(model)
        logging.info("Froze image encoder params: %d", n_frozen)

    # ---- 5) Datasets & loaders
    K = int(cfg["num_classes"])
    ce_weights = parse_ce_weights(cfg["ce_class_weights"], K) if cfg.get("ce_class_weights") else None
    crop_targets = parse_crop_targets(cfg["crop_target_classes"], K) if cfg.get("crop_target_classes") else None
    try:
        apply_norm, norm_mean, norm_std = resolve_normalization(cfg.get("img_normalization"))
    except ValueError as exc:
        raise ValueError(f"Invalid img_normalization preset: {cfg.get('img_normalization')}") from exc
    rot_angle = int(cfg["aug_rot_angle"])
    pad_only = bool(cfg.get("pad_only", False))
    bc_strength = float(cfg.get("bc_strength", 0.2))
    noise_std = float(cfg.get("noise_std", 0.0))
    blur_p = float(cfg.get("blur_p", 0.0))
    blur_k = int(cfg.get("blur_k", 3))
    elastic_p = float(cfg.get("elastic_p", 0.0))
    grid_p = float(cfg.get("grid_p", 0.0))
    gamma_min = float(cfg.get("gamma_min", 0.9))
    gamma_max = float(cfg.get("gamma_max", 1.1))
    if gamma_min <= 0 or gamma_max <= 0:
        raise ValueError("gamma_min and gamma_max must be positive.")
    if gamma_min > gamma_max:
        raise ValueError("gamma_min must be <= gamma_max.")
    use_clahe_cfg = cfg.get("use_clahe", 0)
    if isinstance(use_clahe_cfg, str):
        use_clahe = use_clahe_cfg.strip().lower() in ("1", "true", "yes")
    else:
        use_clahe = bool(use_clahe_cfg)


        

    train_dataset = PairedImageMaskDataset(
        image_dir=Path(cfg["train_images"]),
        mask_dir=Path(cfg["train_masks"]),
        image_size=model.image_size,
        num_classes=K,
        ignore_index=int(cfg["ignore_index"]),
        augment=bool(cfg["augment"]),
        norm_mean=norm_mean,
        norm_std=norm_std,
        apply_norm=apply_norm,
        rotate_angle=rot_angle,
        pad_only=pad_only,
        bc_strength=bc_strength,
        noise_std=noise_std,
        blur_p=blur_p,
        blur_k=blur_k,
        elastic_p=elastic_p,
        grid_p=grid_p,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        use_clahe=use_clahe,
    )
    train_dataset.configure_cropping(enabled=False, crop_frequency=0.0)

    val_dataset = PairedImageMaskDataset(
        image_dir=Path(cfg["val_images"]),
        mask_dir=Path(cfg["val_masks"]),
        image_size=model.image_size,
        num_classes=K,
        ignore_index=int(cfg["ignore_index"]),
        augment=False,
        norm_mean=norm_mean,
        norm_std=norm_std,
        apply_norm=apply_norm,
        rotate_angle=rot_angle,
        pad_only=pad_only,
        bc_strength=bc_strength,
        noise_std=noise_std,
        blur_p=blur_p,
        blur_k=blur_k,
        elastic_p=elastic_p,
        grid_p=grid_p,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        use_clahe=use_clahe,
    )
    val_dataset.configure_cropping(enabled=False, crop_frequency=0.0)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, int(cfg["batch_size"]) // 2),
        shuffle=False,
        num_workers=max(1, int(cfg["num_workers"]) // 2),
        pin_memory=pin_memory,
        drop_last=False,
    )

    # ---- 6) Loss, optimizer, scheduler
    criterion = FocalLossCEAligned(
        gamma=float(cfg["focal_gamma"]),
        weight=ce_weights,
        ignore_index=int(cfg["ignore_index"]),
        reduction='mean'
    )

    base_model = model.module if hasattr(model, "module") else model
    encoder_params, non_encoder_params = [], []
    for name, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        (encoder_params if name.startswith("image_encoder.") else non_encoder_params).append(p)

    opt_groups = []
    if non_encoder_params:
        opt_groups.append({"params": non_encoder_params, "lr": float(cfg["lr"])})
    if encoder_params:
        opt_groups.append({"params": encoder_params, "lr": float(cfg["lr_image_encoder"])})
    optimizer = torch.optim.AdamW(opt_groups, weight_decay=float(cfg["weight_decay"]))

    scheduler = None
    if str(cfg["lr_scheduler"]) == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))

    amp_enabled = bool(cfg["amp"]) and device.type == "cuda"
    scaler = amp.GradScaler(enabled=amp_enabled)
    grad_monitor = GradientMonitor(model)

    # ---- 7) Visualizer & output dir (unique per run)
    visualizer = create_training_visualizer(device=str(device), include_background_as_class=True)
    run_suffix = wandb.run.id if wandb.run else "local"
    output_dir = Path(f"{cfg['output_dir']}_{run_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 8) Train/eval loop
    best_metric = float("-inf")
    best_epoch = 0
    epochs = int(cfg["epochs"])
    grad_clip = float(cfg["grad_clip"])

    try:
        for epoch in range(1, epochs + 1):
            logging.info("Epoch %d/%d", epoch, epochs)

            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                grad_clip=grad_clip,
                num_classes=K,
                ignore_index=int(cfg["ignore_index"]),
                grad_monitor=grad_monitor,
                progress_desc=f"Train {epoch}/{epochs}",
            )

            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=K,
                ignore_index=int(cfg["ignore_index"]),
                progress_desc=f"Val {epoch}/{epochs}",
            )

            if scheduler is not None:
                scheduler.step()

            payload = {}
            payload.update(flatten_metrics("train", train_metrics))
            payload.update(flatten_metrics("val", val_metrics))
            payload["epoch"] = float(epoch)
            payload["lr"] = float(optimizer.param_groups[0]["lr"])
            wandb.log(payload, step=epoch)

            visualize_predictions(
                model=model,
                dataset=val_dataset,
                device=device,
                visualizer=visualizer,
                wandb_wrapper=None,
                epoch=epoch,
                output_dir=output_dir,
            )

            current = float(val_metrics.get("mean_iou", 0.0))
            if current > best_metric:
                best_metric = current
                best_epoch = epoch
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    path=output_dir / "best.pt",
                    scaler=scaler,
                    scheduler=scheduler,
                    best_metric=best_metric,
                )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=output_dir / "last.pt",
                scaler=scaler,
                scheduler=scheduler,
                best_metric=best_metric,
            )

    except torch.cuda.OutOfMemoryError:
        logging.error("⚠️ CUDA OOM encountered — skipping this trial.")
        wandb.alert(title="OOM Trial", text=f"OOM in run {wandb.run.id}, skipping.")
        wandb.finish(exit_code=1)
        return  # abort early, let W&B move to next sweep config

    finally:
        if wandb.run:
            wandb.finish()

    logging.info("Best val mIoU: %.4f at epoch %d", best_metric, best_epoch)
    logging.info("Checkpoints saved to %s", str(output_dir.resolve()))


if __name__ == "__main__":
    train()
    # wandb.agent(sweep_id="2630054-university-of-witwatersrand/sam2_NR206_sweep/4cnr82wr", function=train, count=10)

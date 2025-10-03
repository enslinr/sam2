# semantic_sam2_utils.py

"""
Utilities for Semantic SAM2, including safe checkpoint loading that handles
different numbers of mask tokens/classes.
"""

import torch
import logging
from pathlib import Path
from typing import Optional
import numpy as np


def load_checkpoint_staged_safe(model, ckpt_path: Optional[str]):
    """
    Load checkpoint for Semantic SAM2, handling:
    1. Different number of mask tokens (classes)
    2. SAM2-specific key names
    3. Memory attention/encoder components
    4. PyTorch 2.6+ compatibility
    
    This is a drop-in replacement for the standard _load_checkpoint function
    in sam2.build_sam that handles semantic segmentation models.
    
    Args:
        model: SAM2Semantic model instance
        ckpt_path: Path to checkpoint file (can be None)
    
    Returns:
        None (modifies model in-place)
    """
    if ckpt_path is None:
        logging.info("No checkpoint path provided, skipping checkpoint loading")
        return
    
    checkpoint_path = Path(ckpt_path)
    if not checkpoint_path.exists():
        logging.warning(f"Checkpoint not found: {ckpt_path}")
        return
    
    logging.info(f"Loading checkpoint from: {ckpt_path}")
    
    # ========== Helper Functions ==========
    
    def _strip_module_prefix(sd):
        """Remove 'module.' prefix from DDP checkpoints."""
        if any(k.startswith("module.") for k in sd.keys()):
            return {k.replace("module.", "", 1): v for k, v in sd.items()}
        return sd
    
    def _safe_torch_load(path):
        """Handle PyTorch 2.6+ default change to weights_only=True."""
        try:
            # Try secure mode first
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # Older PyTorch without weights_only
            return torch.load(path, map_location="cpu")
        except Exception:
            # Checkpoint contains non-tensor objects
            return torch.load(path, map_location="cpu", weights_only=False)
    
    def _get_tensor_from_dict(sd, possible_keys):
        """Try multiple possible key names to find a tensor."""
        for key in possible_keys:
            if key in sd:
                return sd[key]
        return None
    
    def _extract_state_dict(checkpoint):
        """Extract state_dict from various checkpoint formats."""
        if isinstance(checkpoint, dict):
            # Training checkpoint with multiple keys
            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                return checkpoint["model"]
            else:
                # Assume the dict itself is the state dict
                return checkpoint
        else:
            # Direct state dict
            return checkpoint
    
    # ========== Semantic Decoder Adaptation ==========
    
    def _adapt_semantic_decoder(model, state_dict):
        """
        Adapt decoder when the number of classes differs between 
        checkpoint and model. Clones head-0 to all class heads.
        
        Returns:
            bool: True if adaptation was performed
        """
        decoder = model.sam_mask_decoder
        device = decoder.mask_tokens.weight.device
        
        # Get number of classes in model vs checkpoint
        K_model = decoder.num_class_tokens
        
        # Try to find mask tokens in checkpoint
        ckpt_mask_tokens = _get_tensor_from_dict(state_dict, [
            "sam_mask_decoder.mask_tokens.weight",
            "mask_decoder.mask_tokens.weight",
        ])
        
        K_ckpt = None
        if ckpt_mask_tokens is not None and ckpt_mask_tokens.ndim == 2:
            K_ckpt = int(ckpt_mask_tokens.shape[0])
        
        logging.info(f"Model has {K_model} class tokens, checkpoint has {K_ckpt}")
        
        # Determine if we need to adapt
        need_adaptation = (K_ckpt is None) or (K_ckpt != K_model)
        
        if need_adaptation:
            logging.info(f"→ Adapting decoder: cloning head-0 to {K_model} classes")
            _clone_head_to_all_classes(decoder, state_dict, K_model, device)
            return True
        
        return False
    
    def _clone_head_to_all_classes(decoder, state_dict, K_model, device):
        """Clone head-0 weights to all class heads."""
        
        # 1. Adapt Mask Tokens
        _adapt_mask_tokens(decoder, state_dict, K_model, device)
        
        # 2. Adapt Hypernetwork MLPs
        _adapt_hypernetwork_mlps(decoder, state_dict, K_model)
        
        # 3. Adapt IoU Prediction Head
        _adapt_iou_head(decoder, state_dict, K_model, device)
    
    def _adapt_mask_tokens(decoder, state_dict, K_model, device):
        """Adapt mask token embeddings."""
        ckpt_tokens = _get_tensor_from_dict(state_dict, [
            "sam_mask_decoder.mask_tokens.weight",
            "mask_decoder.mask_tokens.weight",
        ])
        
        with torch.no_grad():
            if ckpt_tokens is not None and ckpt_tokens.ndim == 2 and ckpt_tokens.shape[0] >= 1:
                # Use first token from checkpoint
                proto = ckpt_tokens[0].to(device)
            else:
                # Use current model's first token
                proto = decoder.mask_tokens.weight.data[0].detach().clone()
            
            # Expand to all classes
            decoder.mask_tokens.weight.data.copy_(
                proto.unsqueeze(0).expand(K_model, -1)
            )
        
        logging.info(f"✓ Adapted mask tokens to {K_model} classes")
    
    def _adapt_hypernetwork_mlps(decoder, state_dict, K_model):
        """Adapt hypernetwork MLPs for all classes."""
        
        # Try to load head-0 from checkpoint
        base_prefixes = [
            "sam_mask_decoder.output_hypernetworks_mlps.0",
            "mask_decoder.output_hypernetworks_mlps.0",
        ]
        
        head0_state = {}
        for prefix in base_prefixes:
            for key, param in state_dict.items():
                if key.startswith(prefix):
                    # Extract layer key: "...mlps.0.layers.0.weight" -> "layers.0.weight"
                    layer_key = key.split(f"{prefix}.")[-1]
                    head0_state[layer_key] = param
        
        # Load head-0 if found
        if head0_state:
            try:
                decoder.output_hypernetworks_mlps[0].load_state_dict(
                    head0_state, strict=False
                )
                logging.info("✓ Loaded hypernetwork head-0 from checkpoint")
            except Exception as e:
                logging.warning(f"Could not load hypernetwork head-0: {e}")
        
        # Clone head-0 to all other heads
        with torch.no_grad():
            head0_state_dict = decoder.output_hypernetworks_mlps[0].state_dict()
            for i in range(1, K_model):
                decoder.output_hypernetworks_mlps[i].load_state_dict(head0_state_dict)
        
        logging.info(f"✓ Cloned hypernetwork head-0 to {K_model} heads")
    
    def _adapt_iou_head(decoder, state_dict, K_model, device):
        """Adapt IoU prediction head for all classes."""
        
        # Get the final linear layer
        last_layer = decoder.iou_prediction_head.layers[-1]
        
        # Try to get checkpoint weights for final layer
        ckpt_weight = _get_tensor_from_dict(state_dict, [
            "sam_mask_decoder.iou_prediction_head.layers.2.weight",
            "mask_decoder.iou_prediction_head.layers.2.weight",
            "sam_mask_decoder.iou_prediction_head.layers.3.weight",  # depth=4
            "mask_decoder.iou_prediction_head.layers.3.weight",
        ])
        
        ckpt_bias = _get_tensor_from_dict(state_dict, [
            "sam_mask_decoder.iou_prediction_head.layers.2.bias",
            "mask_decoder.iou_prediction_head.layers.2.bias",
            "sam_mask_decoder.iou_prediction_head.layers.3.bias",
            "mask_decoder.iou_prediction_head.layers.3.bias",
        ])
        
        with torch.no_grad():
            # Get row-0 from checkpoint or current model
            if (ckpt_weight is not None and ckpt_weight.ndim == 2 and 
                ckpt_weight.shape[0] >= 1 and 
                ckpt_weight.shape[1] == last_layer.weight.shape[1]):
                w0 = ckpt_weight[0:1].to(device)
                b0 = (ckpt_bias[0:1].to(device) if ckpt_bias is not None 
                      else last_layer.bias.data[0:1].clone())
            else:
                w0 = last_layer.weight.data[0:1].detach().clone()
                b0 = last_layer.bias.data[0:1].detach().clone()
            
            # Expand to all classes
            last_layer.weight.data.copy_(w0.expand(K_model, -1))
            last_layer.bias.data.copy_(b0.expand(K_model))
        
        logging.info(f"✓ Adapted IoU head to {K_model} classes")
    
    # ========== Main Loading Logic ==========
    
    try:
        # 1. Load checkpoint file
        checkpoint = _safe_torch_load(str(checkpoint_path))
        logging.info(f"✓ Loaded checkpoint file")
    except Exception as e:
        logging.error(f"✗ Failed to load checkpoint: {e}")
        return
    
    # 2. Extract state dict
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_module_prefix(state_dict)
    
    # 3. Filter to matching shapes
    model_sd = model.state_dict()
    filtered = {}
    mismatched = []
    
    for k, v in state_dict.items():
        if k in model_sd:
            if hasattr(v, "shape") and v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                mismatched.append((k, tuple(getattr(v, "shape", ())), tuple(model_sd[k].shape)))
    
    if mismatched:
        logging.info(f"ℹ  Skipping {len(mismatched)} keys with shape mismatch (showing up to 5):")
        for k, s_ckpt, s_model in mismatched[:5]:
            logging.info(f"   - {k}: ckpt{s_ckpt} vs model{s_model}")
    
    # 4. Load with non-strict mode
    try:
        result = model.load_state_dict(filtered, strict=False)
        
        # Filter out batch norm tracking keys from report
        missing = [k for k in result.missing_keys if "num_batches_tracked" not in k]
        unexpected = [k for k in result.unexpected_keys if "num_batches_tracked" not in k]
        
        if missing or unexpected:
            logging.info(f"ℹ  load_state_dict(strict=False):")
            if missing:
                logging.info(f"   missing keys: {len(missing)}")
                if len(missing) <= 10:
                    for k in missing:
                        logging.info(f"      - {k}")
            if unexpected:
                logging.info(f"   unexpected keys: {len(unexpected)}")
                if len(unexpected) <= 10:
                    for k in unexpected:
                        logging.info(f"      - {k}")
        else:
            logging.info("✓ Loaded checkpoint with strict=False (all keys matched)")
    
    except Exception as e:
        logging.warning(f"⚠  Full model load failed: {e}")
        logging.info("→ Attempting component-wise loading...")
        
        # Try loading components separately
        components = [
            ("image_encoder", model.image_encoder),
            ("sam_prompt_encoder", model.sam_prompt_encoder),
            ("sam_mask_decoder", model.sam_mask_decoder),
            ("memory_attention", model.memory_attention),
            ("memory_encoder", model.memory_encoder),
        ]
        
        for name, module in components:
            try:
                comp_state = {
                    k.replace(f"{name}.", "", 1): v
                    for k, v in state_dict.items()
                    if k.startswith(f"{name}.")
                }
                if comp_state:
                    module.load_state_dict(comp_state, strict=False)
                    logging.info(f"✓ Loaded {name}")
                else:
                    logging.info(f"⚠  No {name} weights in checkpoint")
            except Exception as ce:
                logging.warning(f"⚠  Failed to load {name}: {ce}")
    
    # 5. Adapt decoder for semantic segmentation
    try:
        was_adapted = _adapt_semantic_decoder(model, state_dict)
        if was_adapted:
            logging.info("✓ Decoder adapted for semantic segmentation")
        else:
            logging.info("ℹ  No decoder adaptation needed (classes match)")
    except Exception as e:
        logging.warning(f"⚠  Decoder adaptation failed: {e}")
        import traceback
        traceback.print_exc()
    
    logging.info("✓ Checkpoint loading complete")


def get_semantic_sam2_configs():
    """
    Return a dictionary of available semantic SAM2 configurations.
    Useful for documentation and validation.
    """
    return {
        "hiera_tiny": {
            "config": "sam2.1_hiera_t_semantic",
            "base_config": "sam2.1_hiera_t",
            "embed_dim": 96,
            "num_heads": 1,
        },
        "hiera_small": {
            "config": "sam2.1_hiera_s_semantic",
            "base_config": "sam2.1_hiera_s",
            "embed_dim": 96,
            "num_heads": 1,
        },
        "hiera_base_plus": {
            "config": "sam2.1_hiera_b+_semantic",
            "base_config": "sam2.1_hiera_b+",
            "embed_dim": 112,
            "num_heads": 2,
        },
        "hiera_large": {
            "config": "sam2.1_hiera_l_semantic",
            "base_config": "sam2.1_hiera_l",
            "embed_dim": 144,
            "num_heads": 2,
        },
    }


def print_model_info(model):
    """Print useful information about a semantic SAM2 model."""
    print("=" * 60)
    print("Semantic SAM2 Model Information")
    print("=" * 60)
    
    # Basic info
    print(f"Model type: {type(model).__name__}")
    print(f"Number of classes: {model.num_classes}")
    print(f"Image size: {model.image_size}")
    print(f"Device: {model.device}")
    
    # Decoder info
    decoder = model.sam_mask_decoder
    print(f"\nMask Decoder:")
    print(f"  Num class tokens: {decoder.num_class_tokens}")
    print(f"  Num mask tokens: {decoder.num_mask_tokens}")
    print(f"  Transformer dim: {decoder.transformer_dim}")
    
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    
    print("=" * 60)


def verify_semantic_output(masks, num_classes):
    """
    Verify that the output from semantic SAM2 has the expected shape.
    
    Args:
        masks: Output masks from predictor
        num_classes: Expected number of classes
    
    Returns:
        bool: True if valid, False otherwise
    """
    if masks.shape[0] != num_classes:
        logging.error(
            f"Expected {num_classes} classes, got {masks.shape[0]} masks"
        )
        return False
    
    logging.info(f"✓ Output shape verified: {masks.shape}")
    return True
# semantic_sam2_components.py

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import MLP


class SemanticMaskDecoder(MaskDecoder):
    """
    Extended mask decoder for semantic segmentation with multiple class outputs.
    Similar to your SAM1 MaskDecoderSemantic but adapted for SAM2.
    """
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_class_tokens: int = 11,  # Number of semantic classes
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Initialize semantic mask decoder.
        
        Args:
            num_class_tokens: Number of semantic classes (K)
            Other args are same as base MaskDecoder
        """
        # Initialize parent with num_multimask_outputs = num_class_tokens - 1
        # because parent expects num_multimask_outputs, but we'll override the tokens
        super().__init__(
            transformer_dim=transformer_dim,
            transformer=transformer,
            num_multimask_outputs=3,  # Will be overridden
            activation=activation,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
            use_high_res_features=use_high_res_features,
            iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
            dynamic_multimask_via_stability=dynamic_multimask_via_stability,
            dynamic_multimask_stability_delta=dynamic_multimask_stability_delta,
            dynamic_multimask_stability_thresh=dynamic_multimask_stability_thresh,
            pred_obj_scores=pred_obj_scores,
            pred_obj_scores_mlp=pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=use_multimask_token_for_obj_ptr,
        )
        
        self.num_class_tokens = num_class_tokens
        self.num_mask_tokens = num_class_tokens
        
        # Replace mask tokens with semantic class tokens
        self.mask_tokens = nn.Embedding(num_class_tokens, transformer_dim)
        
        # Replace hypernetwork MLPs for all class tokens
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(num_class_tokens)
        ])
        
        # Update IoU head to predict IoU for each class
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            num_class_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        sam_tokens_out = mask_tokens_out

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits


class ClassTaggedPromptEncoder(PromptEncoder):
    """
    Extended prompt encoder that can handle class-specific prompts.
    Similar to your SAM1 version.
    """
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        num_classes: int = 11,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=input_image_size,
            mask_in_chans=mask_in_chans,
            activation=activation,
        )
        self.num_classes = num_classes
        
        # Optional: Add class token embeddings if you want to condition on class
        self.class_embeddings = nn.Embedding(num_classes, embed_dim)


class SAM2Semantic(SAM2Base):
    """
    Semantic SAM2 that outputs per-class masks instead of generic object masks.
    Extends SAM2Base to use semantic decoder.
    """
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_classes: int = 11,
        **kwargs
    ):
        # Store num_classes before calling super
        self.num_classes = num_classes
        
        # Call parent init (this will call _build_sam_heads)
        super().__init__(
            image_encoder=image_encoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            **kwargs
        )
    
    def _build_sam_heads(self):
        """Override to build semantic decoder instead of standard decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        
        # Build semantic prompt encoder
        self.sam_prompt_encoder = ClassTaggedPromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
            num_classes=self.num_classes,
        )
        
        # Build semantic mask decoder
        from sam2.modeling.sam.transformer import TwoWayTransformer
        
        self.sam_mask_decoder = SemanticMaskDecoder(
            num_class_tokens=self.num_classes,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        
        # Object pointer projection
        if self.use_obj_ptrs_in_encoder:
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
            
        if self.proj_tpos_enc_in_obj_ptrs:
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()


# # For training
# from sam2.training.sam2 import SAM2Train


# class SAM2SemanticTrain(SAM2Train):
#     """
#     Training wrapper for semantic SAM2.
#     Inherits all training logic but uses semantic base.
#     """
#     def __init__(
#         self,
#         image_encoder,
#         memory_attention=None,
#         memory_encoder=None,
#         num_classes: int = 11,
#         **kwargs
#     ):
#         # Temporarily store num_classes
#         self._num_classes_init = num_classes
        
#         # Call parent __init__ which will eventually call _build_sam_heads
#         # We need to ensure our num_classes is available there
#         super(SAM2Train, self).__init__(  # Skip SAM2Train.__init__, go to SAM2Base
#             image_encoder=image_encoder,
#             memory_attention=memory_attention,
#             memory_encoder=memory_encoder,
#             **kwargs
#         )
        
#         # Store num_classes as instance attribute
#         self.num_classes = num_classes
        
#         # Now apply SAM2Train-specific initialization
#         # (Copy relevant parts from SAM2Train.__init__)
#         from sam2.training.sam2 import SAM2Train as BaseSAM2Train
        
#         # Copy training-specific attributes
#         for attr in dir(BaseSAM2Train):
#             if not attr.startswith('_') and attr not in ['forward', '__init__']:
#                 try:
#                     setattr(self, attr, getattr(BaseSAM2Train, attr))
#                 except:
#                     pass
    
#     def _build_sam_heads(self):
#         """Use semantic decoder for training."""
#         SAM2Semantic._build_sam_heads(self)
# semantic_sam2_components.py

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import LayerNorm2d, MLP


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
        hypernet_output_dim: Optional[int] = None
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


# Increasing dims
        self.hypernet_output_dim = (
            transformer_dim // 8 if hypernet_output_dim is None else hypernet_output_dim
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4,
                self.hypernet_output_dim,
                kernel_size=2,
                stride=2,
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )
            # project widened decoder features down to 32-ch before adding feat_s0,
            # then lift them back up for the hyper-networks
            self.upscale_residual_proj = nn.Conv2d(
                self.hypernet_output_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.upscale_post_add_proj = nn.Conv2d(
                transformer_dim // 8, self.hypernet_output_dim, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, self.hypernet_output_dim, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )
# Increasing dims end

        # # Replace hypernetwork MLPs for all class tokens
        # self.output_hypernetworks_mlps = nn.ModuleList([
        #     MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #     for _ in range(num_class_tokens)
        # ])
        
        # # Update IoU head to predict IoU for each class
        # self.iou_prediction_head = MLP(
        #     transformer_dim,
        #     iou_head_hidden_dim,
        #     num_class_tokens,
        #     iou_head_depth,
        #     sigmoid_output=iou_prediction_use_sigmoid,
        # )

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


    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            # dc1, ln1, act1, dc2, act2 = self.output_upscaling
            # feat_s0, feat_s1 = high_res_features
            # upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            # upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = dc1(src)
            upscaled_embedding = ln1(upscaled_embedding + feat_s1)
            upscaled_embedding = act1(upscaled_embedding)
            upscaled_embedding = dc2(upscaled_embedding)
            fusion = self.upscale_residual_proj(upscaled_embedding) + feat_s0
            fusion = act2(fusion)
            upscaled_embedding = self.upscale_post_add_proj(fusion)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits






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
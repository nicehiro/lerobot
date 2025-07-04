import math
from typing import Optional

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)

from lerobot.common.policies.vla.head import DiffusionActionHead, FlowMatchingActionHead


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class FeaturePoolingBridge(nn.Module):
    """Feature pooling bridge for connecting vision-language backbone to action expert."""

    def __init__(self, input_dim: int, output_dim: int, pooling_type: str = "attention"):
        super().__init__()
        self.pooling_type = pooling_type
        self.input_dim = input_dim
        self.output_dim = output_dim

        if pooling_type == "attention":
            self.attention_weights = nn.Linear(input_dim, 1)
            self.projection = nn.Linear(input_dim, output_dim)
        elif pooling_type in ["max", "mean"]:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [batch_size, seq_len, feature_dim]
            attention_mask: [batch_size, seq_len] - mask for valid tokens
        Returns:
            pooled_features: [batch_size, output_dim]
        """
        if self.pooling_type == "max":
            if attention_mask is not None:
                features = features.masked_fill(~attention_mask.unsqueeze(-1), float("-inf"))
            pooled = torch.max(features, dim=1)[0]
        elif self.pooling_type == "mean":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(features)
                features = features * mask_expanded
                pooled = features.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = torch.mean(features, dim=1)
        elif self.pooling_type == "attention":
            attention_scores = self.attention_weights(features).squeeze(-1)
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(~attention_mask, float("-inf"))
            attention_weights = torch.softmax(attention_scores, dim=1)
            pooled = torch.sum(features * attention_weights.unsqueeze(-1), dim=1)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        return self.projection(pooled)


class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        head_config=None,
    ):
        super().__init__()

        # Validate head_config
        if head_config is None:
            raise ValueError("head_config is required")

        if load_vlm_weights:
            print(f"Loading  {model_id} weights ...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.vlm_config = config

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        self.head_config = head_config
        self._init_dual_system_components()

        self.set_requires_grad()

    def _init_dual_system_components(self):
        """Initialize dual system VLA components."""
        self.vlm_hidden_size = self.vlm_config.text_config.hidden_size
        self.feature_bridge = FeaturePoolingBridge(
            input_dim=self.vlm_hidden_size,
            output_dim=self.head_config.expert_hidden_size,
            pooling_type=self.head_config.pooling_type,
        )
        self.action_in_proj = nn.Linear(self.head_config.max_action_dim, self.head_config.expert_hidden_size)
        self.state_proj = nn.Linear(self.head_config.max_state_dim, self.head_config.expert_hidden_size)

        # Initialize action head based on configuration
        if self.head_config.head_type == "diffusion":
            self.action_head = DiffusionActionHead(self.head_config)
        elif self.head_config.head_type == "flow_matching":
            self.action_head = FlowMatchingActionHead(self.head_config)
        else:
            raise ValueError(f"Unsupported head type: {self.head_config.head_type}")

    def get_vlm_model(self):
        return self.vlm.model

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            last_layers = [self.config.text_config.num_hidden_layers - 1]
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")

            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False

    def embed_image(self, image: torch.Tensor):
        patch_attention_mask = None
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images and language tokens to prepare for VLM processing.

        Args:
            images: List of image tensors
            img_masks: List of image masks
            lang_tokens: Language token tensor
            lang_masks: Language mask tensor
            state: State tensor

        Returns:
            Tuple of (embeddings, padding_masks, attention_masks)
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.embed_image(img)
            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs
        # Process language tokens
        lang_emb = self.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        # Process state
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device
        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        # Set attention masks so that image and language inputs do not attend to state
        att_masks += [1] * states_seq_len
        # Concatenate all embeddings
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]
        bsize = embs.shape[0]
        att_masks = att_masks.expand(bsize, -1)
        return embs, pad_masks, att_masks

    def compute_loss(self, images, img_masks, lang_tokens, lang_masks, state, actions):
        """Compute loss for action prediction.

        Args:
            images: List of image tensors
            img_masks: List of image masks
            lang_tokens: Language token tensor
            lang_masks: Language mask tensor
            state: State tensor
            actions: Ground truth actions

        Returns:
            Loss tensor
        """
        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        pooled_features = self.feature_bridge(prefix_embs, prefix_pad_masks)
        loss = self.action_head.compute_loss(actions, pooled_features)
        return loss

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, num_inference_steps=10
    ):
        """Sample actions during inference.

        Args:
            images: List of image tensors
            img_masks: List of image masks
            lang_tokens: Language token tensor
            lang_masks: Language mask tensor
            state: State tensor
            noise: Optional noise tensor
            num_inference_steps: Number of inference steps

        Returns:
            Sampled actions
        """
        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        pooled_features = self.feature_bridge(prefix_embs, prefix_pad_masks)
        return self.action_head.sample(pooled_features, num_inference_steps=num_inference_steps)

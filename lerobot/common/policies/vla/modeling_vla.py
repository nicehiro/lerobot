"""
Dual-system Vision-Language-Action (VLA) policy implementation.
"""

import math
import os
import re
from collections import deque

import safetensors
import torch
from torch import Tensor
from transformers import AutoProcessor

from lerobot.common.constants import ACTION, OBS_STATE
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import populate_queues
from lerobot.common.policies.vla.configuration_vla import VLAConfig
import torch.nn.functional as F

from lerobot.common.policies.vla.smolvlm_with_expert import SmolVLMWithExpertModel


def pad_vector(vector, max_len, pad_value=0):
    """Pad vector to max_len."""
    if vector.shape[-1] >= max_len:
        return vector[..., :max_len]
    pad_shape = list(vector.shape)
    pad_shape[-1] = max_len - vector.shape[-1]
    padding = torch.full(pad_shape, pad_value, dtype=vector.dtype, device=vector.device)
    return torch.cat([vector, padding], dim=-1)


def resize_with_pad(img, height, width, pad_value=0):
    """Resize image with padding to maintain aspect ratio."""
    _, _, h, w = img.shape
    scale = min(height / h, width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # Pad
    pad_h = height - new_h
    pad_w = width - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
    return img

# Matches ".soNNN", optionally followed by "-something", up to the "_buffer_" marker
_VARIANT_RE = re.compile(r"\.so\d+(?:-[\w]+)?_buffer_")


def canonicalise(k: str) -> str:
    """Remove dataset-variant markers from normalisation-buffer key."""
    return _VARIANT_RE.sub(".buffer_", k)


def standardise_state_dict(
    checkpoint: dict[str, torch.Tensor], ref_keys: set[str], *, verbose: bool = True
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Re-keys checkpoint to match reference key set."""
    out, collisions, unmatched = {}, {}, []

    for k, v in checkpoint.items():
        canon = canonicalise(k)
        if canon in ref_keys:
            if canon in out:
                collisions.setdefault(canon, []).append(k)
            else:
                out[canon] = v
        else:
            unmatched.append(k)

    if verbose:
        for canon, variants in collisions.items():
            print(f"[standardise_state_dict] '{canon}'  ←  {variants}")
        if unmatched:
            print(f"[standardise_state_dict] kept {len(unmatched)} unmatched keys")

    out.update({k: checkpoint[k] for k in unmatched})
    return out, unmatched


def rename_checkpoint_keys(checkpoint: dict, rename_str: str):
    """Renames keys in a checkpoint dictionary based on the given rename string."""
    rename_dict = dict(pair.split("//") for pair in rename_str.split(","))

    new_checkpoint = {}
    for k, v in checkpoint.items():
        for old_key, new_key in rename_dict.items():
            if old_key in k:
                k = k.replace(old_key, new_key)
        new_checkpoint[k] = v
    return new_checkpoint


def load_vla(
    model: torch.nn.Module,
    filename: str | os.PathLike,
    *,
    device: str = "cpu",
    checkpoint_keys_mapping: str = "",
) -> torch.nn.Module:
    state_dict = safetensors.torch.load_file(filename, device=device)

    if checkpoint_keys_mapping and "//" in checkpoint_keys_mapping:
        state_dict = rename_checkpoint_keys(state_dict, checkpoint_keys_mapping)

    state_dict, _ = standardise_state_dict(state_dict, set(model.state_dict().keys()))

    # Don't overwrite normalization parameters
    norm_keys = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(norm_keys)}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if not all(key.startswith(norm_keys) for key in missing) or unexpected:
        raise RuntimeError(f"VLA {len(missing)} missing / {len(unexpected)} unexpected keys")

    return model


def pad_tensor(tensor, max_len, pad_value=0):
    """Efficiently pads a tensor along sequence dimension to match max_len."""
    b, d = tensor.shape[:2]
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor
    return padded_tensor


class VLAPolicy(PreTrainedPolicy):
    """Dual-system VLA policy for training and inference within LeRobot."""

    config_class = VLAConfig
    name = "vla"

    def __init__(self, config: VLAConfig, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = AutoProcessor.from_pretrained(self.config.vlm_model_name).tokenizer

        # Initialize the dual system VLA model with the integrated SmolVLMWithExpertModel
        self.model = SmolVLMWithExpertModel(
            model_id=config.vlm_model_name,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=False,  # We need VLM features for dual system
            load_vlm_weights=config.load_vlm_weights,
            head_config=config,  # Pass VLA config for dual system components
        )
        
        # Mark this as a dual system VLA for the training script
        self.dual_system_vla = True

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    @classmethod
    def _load_as_safetensor(
        cls,
        model: "VLAPolicy",
        model_file: str,
        map_location: str,
        strict: bool,
    ):
        safetensors.torch.load_model(model, model_file, strict=strict, device=map_location)
        return load_vla(
            model,
            model_file,
            device=map_location,
            checkpoint_keys_mapping="model._orig_mod.//model.",
        )

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        batch = self.normalize_inputs(batch)

        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            for k in batch:
                if k in self._queues:
                    batch[k] = torch.stack(list(self._queues[k]), dim=1)
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)

            actions = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, num_inference_steps=self.config.num_inference_steps
            )
            # Unpad actions to original action dimension
            if hasattr(self.config, 'action_feature') and hasattr(self.config.action_feature, 'shape'):
                original_action_dim = self.config.action_feature.shape[0]
                actions = actions[:, :, :original_action_dim]
            else:
                # Fallback: use the actual action dimension from dataset
                original_action_dim = batch[ACTION].shape[-1] if ACTION in batch else actions.shape[-1]
                actions = actions[:, :, :original_action_dim]

            actions = self.unnormalize_outputs({"action": actions})["action"]

            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")

        loss = self.model.compute_loss(images, img_masks, lang_tokens, lang_masks, state, actions)
        loss_dict = {'total_loss': loss.item()}

        # Apply action padding masks if available
        if actions_is_pad is not None:
            # The loss computation now handles masking internally in action head
            # We can add episode boundary information to loss_dict for monitoring
            in_episode_bound = ~actions_is_pad
            loss_dict["in_episode_ratio"] = in_episode_bound.float().mean().item()

        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply VLA preprocessing to images."""
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_STATE].device
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks]

        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch[OBS_STATE].shape[0])]

        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer(
            tasks,
            padding=self.config.pad_language_to,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action to match the expected action dimension."""
        actions = batch[ACTION]
        if actions.ndim == 2:
            # Add sequence dimension if missing
            actions = actions.unsqueeze(1)
        # Pad to max action dimension
        actions = pad_vector(actions, self.config.max_action_dim)
        return actions

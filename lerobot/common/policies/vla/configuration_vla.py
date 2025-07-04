from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("vla")
@dataclass
class VLAConfig(PreTrainedConfig):
    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 8
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Dimensions (shorter vectors will be padded)
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Language processing
    tokenizer_max_length: int = 48
    pad_language_to: str = "longest"  # "max_length"

    # VLM backbone
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = True
    freeze_vision_encoder: bool = True

    pooling_type: str = "attention"  # "max", "mean", "attention"

    # Action Head parameters
    head_type: str = "diffusion"  # "diffusion" or "flow_matching"
    condition_dropout_prob: float = 0.1  # Dropout probability for classifier-free guidance
    beta_schedule: str = "cosine"  # Noise schedule: "linear" or "cosine"
    diffusion_prediction_type: str = "epsilon"  # "epsilon" (noise) or "sample" (clean actions)
    num_inference_steps: int = 10  # Number of sampling steps during inference

    # DiT model configuration
    dit_model_name: str = "DiT-B"  # "DiT-S", "DiT-B", "DiT-L"
    expert_hidden_size: int = 512
    
    # Additional VLA-specific attributes
    empty_cameras: int = 2  # Number of empty cameras to handle

    # Action prediction
    num_diffusion_steps: int = 1000
    num_steps: int = 10

    # Time embedding
    min_period: float = 4e-3
    max_period: float = 4.0

    # Training settings
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

    def validate_features(self) -> None:
        """Validate VLA-specific feature configuration."""
        # Get image and action features from the config
        if hasattr(self, 'features') and self.features:
            image_features = [key for key, feature in self.features.items() 
                             if hasattr(feature, 'shape') and len(feature.shape) > 2]
            action_features = [key for key, feature in self.features.items() 
                              if key.startswith('action')]
            
            if not image_features:
                # Set default image features if none found
                self.image_features = ['observation.image']
            else:
                self.image_features = image_features
                
            if action_features:
                self.action_feature = self.features[action_features[0]]
            else:
                self.action_feature = None
        else:
            # Set defaults if features not available
            self.image_features = ['observation.image']
            self.action_feature = None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
        
    @property
    def input_features(self) -> list:
        """Features used as inputs to the policy."""
        if hasattr(self, 'features'):
            return [key for key in self.features.keys() if not key.startswith('action')]
        return ['observation.image', 'observation.state']
        
    @property
    def output_features(self) -> list:
        """Features used as outputs from the policy."""
        if hasattr(self, 'features'):
            return [key for key in self.features.keys() if key.startswith('action')]
        return ['action']

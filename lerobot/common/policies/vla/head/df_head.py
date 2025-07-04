import torch
import torch.nn.functional as F  # noqa: N812

import lerobot.common.policies.vla.diffusion.gaussian_diffusion as gd
from lerobot.common.policies.vla.head.action_head import ActionHead, create_diffusion, sample_beta
from lerobot.common.policies.vla.head.dit import DiT_models


class DiffusionActionHead(ActionHead):
    """
    Diffusion-based action head for action prediction.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = config.num_diffusion_steps
        self.noise_schedule = config.beta_schedule
        self.diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=self.noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False,
        )
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False

        model_name = config.dit_model_name
        self.net = DiT_models[model_name](
            token_size=config.expert_hidden_size,
            in_channels=config.max_action_dim,
            class_dropout_prob=config.condition_dropout_prob,
            learn_sigma=learn_sigma,
            future_action_window_size=config.n_action_steps,
            past_action_window_size=0,
        )

    def forward(self, x_t, t, z):
        noise_predict = self.net(x_t, t, z)
        return noise_predict

    def compute_loss(self, actions, conditions, action_masks=None):
        """Compute diffusion loss for action prediction.

        Args:
            actions: Ground truth actions [batch_size, seq_len, action_dim]
            conditions: Condition embeddings [batch_size, condition_dim]
            action_masks: Optional mask for actions

        Returns:
            Loss tensor
        """
        # Use improved noise sampling (better for training stability)
        action_noise = torch.randn_like(actions)
        
        # Use uniform time sampling for better coverage
        time = torch.randint(0, self.diffusion.num_timesteps, (actions.size(0),), device=actions.device)
        # sample x_t from x
        x_t = self.diffusion.q_sample(actions, time, action_noise)

        # Expand conditions to match sequence length
        if conditions.ndim == 2:
            conditions = conditions.unsqueeze(1)  # [batch_size, 1, condition_dim]

        noise_predict = self.forward(x_t, time, conditions)

        if action_masks is not None:
            losses = F.mse_loss(noise_predict, action_noise, reduction="none")
            losses = losses * action_masks.unsqueeze(-1)
            return losses.mean()
        else:
            return F.mse_loss(noise_predict, action_noise)

    def sample(self, conditions, num_inference_steps=None, generator=None):
        """Sample actions using DDIM sampling.

        Args:
            conditions: Condition embeddings [batch_size, condition_dim]
            num_inference_steps: Number of inference steps
            generator: Random number generator

        Returns:
            Sampled actions [batch_size, seq_len, action_dim]
        """
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps

        # Create DDIM sampler if not exists
        if self.ddim_diffusion is None:
            self.create_ddim(num_inference_steps)

        batch_size = conditions.size(0)
        seq_len = self.config.n_action_steps
        action_dim = self.config.max_action_dim

        # Sample initial noise
        if generator is not None:
            noise = torch.randn(
                batch_size, seq_len, action_dim, device=conditions.device, generator=generator
            )
        else:
            noise = torch.randn(batch_size, seq_len, action_dim, device=conditions.device)

        # Expand conditions to match sequence length
        if conditions.ndim == 2:
            conditions = conditions.unsqueeze(1)  # [batch_size, 1, condition_dim]

        # Sample using DDIM
        def model_fn(x, t):
            return self.forward(x, t, conditions)

        sample = self.ddim_diffusion.ddim_sample_loop(
            model_fn,
            shape=(batch_size, seq_len, action_dim),
            noise=noise,
            clip_denoised=True,
            device=conditions.device,
        )

        return sample

    def sample_noise(self, shape, device):
        """Sample noise tensor with improved stability."""
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def create_ddim(self, ddim_step=10):
        """Create DDIM diffusion sampler."""
        self.ddim_diffusion = create_diffusion(
            timestep_respacing="ddim" + str(ddim_step),
            noise_schedule=self.noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False,
        )
        return self.ddim_diffusion

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.common.policies.vla.head.action_head import ActionHead
from lerobot.common.policies.vla.head.dit import DiT_models


class FlowMatchingActionHead(ActionHead):
    """
    Flow matching-based action head for action prediction.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Set DiT model parameters from config
        model_name = config.dit_model_name
        self.net = DiT_models[model_name](
            token_size=config.expert_hidden_size,
            in_channels=config.max_action_dim,
            class_dropout_prob=config.condition_dropout_prob,
            learn_sigma=False,
            future_action_window_size=config.n_action_steps,
            past_action_window_size=0,
        )

    def forward(self, x_t, t, conditions):
        """Forward pass through the flow matching model.

        Args:
            x_t: Noisy actions at time t [batch_size, seq_len, action_dim]
            t: Timestep [batch_size]
            conditions: Condition embeddings [batch_size, 1, condition_dim]

        Returns:
            Velocity field prediction [batch_size, seq_len, action_dim]
        """
        # DiT model already outputs the correct action dimension
        v_t = self.net(x_t, t, conditions)
        return v_t

    def compute_loss(self, actions, conditions, action_masks=None):
        """Compute flow matching loss.

        Args:
            actions: Ground truth actions [batch_size, seq_len, action_dim]
            conditions: Condition embeddings [batch_size, condition_dim]
            action_masks: Optional mask for actions

        Returns:
            Loss tensor
        """
        # Use improved noise sampling
        action_noises = torch.randn_like(actions)

        # Use more stable time sampling for flow matching
        time = torch.rand(actions.shape[0], device=actions.device)

        # Flow matching interpolation: x_t = t * x_1 + (1 - t) * x_0
        # where x_0 is noise and x_1 is data
        time_expanded = time[:, None, None]
        x_t = time_expanded * actions + (1 - time_expanded) * action_noises

        # Target velocity field: u_t = x_1 - x_0
        u_t = actions - action_noises

        # Expand conditions to match sequence length
        if conditions.ndim == 2:
            conditions = conditions.unsqueeze(1)  # [batch_size, 1, condition_dim]

        # Get velocity field prediction
        v_t = self.forward(x_t, time, conditions)

        if action_masks is not None:
            losses = F.mse_loss(v_t, u_t, reduction="none")
            losses = losses * action_masks.unsqueeze(-1)
            return losses.mean()
        else:
            return F.mse_loss(v_t, u_t)

    def sample(self, conditions, num_inference_steps=None, generator=None):
        """Sample actions using flow matching with Euler integration.

        Args:
            conditions: Condition embeddings [batch_size, condition_dim]
            num_inference_steps: Number of integration steps
            generator: Random number generator

        Returns:
            Sampled actions [batch_size, seq_len, action_dim]
        """
        if num_inference_steps is None:
            num_inference_steps = getattr(self.config, 'num_inference_steps', 10)

        batch_size = conditions.size(0)
        seq_len = self.config.n_action_steps
        action_dim = self.config.max_action_dim

        if generator is not None:
            x = torch.randn(batch_size, seq_len, action_dim,
                          device=conditions.device, generator=generator)
        else:
            x = torch.randn(batch_size, seq_len, action_dim, device=conditions.device)

        if conditions.ndim == 2:
            conditions = conditions.unsqueeze(1)  # [batch_size, 1, condition_dim]

        dt = 1.0 / num_inference_steps
        for i in range(num_inference_steps):
            t = torch.full((batch_size,), i * dt, device=conditions.device)
            v_t = self.forward(x, t, conditions)
            x = x + dt * v_t

        return x

    def sample_noise(self, shape, device):
        """Sample noise tensor with improved stability."""
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        """Sample time with uniform distribution for better training."""
        # Uniform sampling is more stable than beta distribution for flow matching
        return torch.rand(bsize, dtype=torch.float32, device=device)

import math
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

import lerobot.common.policies.vla.diffusion.gaussian_diffusion as gd
from lerobot.common.policies.vla.diffusion.respace import SpacedDiffusion, space_timesteps
from lerobot.common.policies.vla.dit import DiT_models


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=(
            (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
    )


class ActionHead(nn.Module):
    """
    Action Head for action prediction.
    """

    def __init__(self, config):
        super().__init__()

    def forward(self, **kwargs):
        pass

    def compute_loss(self, prefix_embs, actions, action_masks=None):
        pass

    def sample(self, conditions, num_inference_steps=None, generator=None):
        pass

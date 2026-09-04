from typing import Literal

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

_DIFFUSION_TRAIN_TIMESTEP_COUNT = 100
_DDIM_INFERENCE_TIMESTEP_COUNT = 8


def create_forward_noise_scheduler() -> DDPMScheduler:
    """The schedule that adds noise to the actions during training.
    Both DDPM and DDIM samplers add noise identically, so the choice of DDPM here is arbitrary.
    """
    return _create_ddpm_scheduler()


def create_sampler(*, name: Literal["ddim", "ddpm"]) -> DDIMScheduler | DDPMScheduler:
    match name:
        case "ddim":
            return _create_ddim_scheduler()
        case "ddpm":
            return _create_ddpm_scheduler()


def _create_ddpm_scheduler() -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=_DIFFUSION_TRAIN_TIMESTEP_COUNT,
        beta_schedule="squaredcos_cap_v2",
        # clip output to [-1, 1] for stability
        clip_sample=True,
        # network predicts noise, not denoised action
        prediction_type="epsilon",
    )


def _create_ddim_scheduler() -> DDIMScheduler:
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=_DIFFUSION_TRAIN_TIMESTEP_COUNT,
        beta_schedule="squaredcos_cap_v2",
        # clip output to [-1, 1] for stability
        clip_sample=True,
        # network predicts noise, not denoised action
        prediction_type="epsilon",
    )
    noise_scheduler.set_timesteps(num_inference_steps=_DDIM_INFERENCE_TIMESTEP_COUNT)
    return noise_scheduler

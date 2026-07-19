from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

_DIFFUSION_TRAIN_TIMESTEP_COUNT = 100
_DDIM_INFERENCE_TIMESTEP_COUNT = 8


def get_ddpm_scheduler() -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=_DIFFUSION_TRAIN_TIMESTEP_COUNT,
        # squared-cosine schedule gave the best results in experiments
        beta_schedule="squaredcos_cap_v2",
        # clip output to [-1, 1] for stability
        clip_sample=True,
        # network predicts noise, not denoised action
        prediction_type="epsilon",
    )


def get_ddim_scheduler() -> DDIMScheduler:
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=_DIFFUSION_TRAIN_TIMESTEP_COUNT,
        # squared-cosine schedule gave the best results in experiments
        beta_schedule="squaredcos_cap_v2",
        # clip output to [-1, 1] for stability
        clip_sample=True,
        # network predicts noise, not denoised action
        prediction_type="epsilon",
    )
    noise_scheduler.set_timesteps(num_inference_steps=_DDIM_INFERENCE_TIMESTEP_COUNT)
    return noise_scheduler

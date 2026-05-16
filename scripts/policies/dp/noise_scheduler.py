from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def get_noise_scheduler(diffusion_timestep_count: int) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=diffusion_timestep_count,
        # squared-cosine schedule gave the best results in experiments
        beta_schedule="squaredcos_cap_v2",
        # clip output to [-1, 1] for stability
        clip_sample=True,
        # network predicts noise, not denoised action
        prediction_type="epsilon",
    )

from gym_pusht.envs import PushTEnv

# Push T runs at 10 actions per second, which is slow to watch.
PLAYBACK_SPEED = 3.5


def get_video_fps() -> int:
    actions_per_second = PushTEnv.metadata["render_fps"]
    if not isinstance(actions_per_second, int):
        raise TypeError(
            f"Expected an integer action rate, got {type(actions_per_second).__name__}"
        )
    return round(actions_per_second * PLAYBACK_SPEED)

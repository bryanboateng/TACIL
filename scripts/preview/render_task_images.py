from dataclasses import dataclass
from pathlib import Path

import gym_pusht  # noqa: F401, registers gym_pusht/PushT-v0
import gymnasium
import tyro
from imageio import v2 as imageio

from scripts.environments.registry import EnvironmentName, create_environment
from scripts.environments.render_settings import RenderSettings
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass(frozen=True)
class Config:
    """Render one still image per task at its initial state, without the video HUD.

    Push-T is drawn with gym_pusht's own colors instead of the condition colormap.
    """

    output_directory: Path = OUTPUT_DIRECTORY_PATH / "task_images"
    seed: int = 30_000
    size: int = 1_000
    environments: tuple[EnvironmentName, ...] = tuple(EnvironmentName)


def main() -> None:
    config = tyro.cli(f=Config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    for name in config.environments:
        gym_environment = _create_gym_environment(name=name, size=config.size)
        try:
            gym_environment.reset(seed=config.seed)
            frame = gym_environment.render()
        finally:
            gym_environment.close()
        path = config.output_directory / f"{name.value}.png"
        imageio.imwrite(uri=path, im=frame)
        logger.info("Wrote %s", path)


# ---- Private --------------------------------------------


def _create_gym_environment(*, name: EnvironmentName, size: int) -> gymnasium.Env:
    if name is EnvironmentName.PUSH_T:
        return gymnasium.make(
            id="gym_pusht/PushT-v0",
            render_mode="rgb_array",
            visualization_width=size,
            visualization_height=size,
        )
    environment = create_environment(name=name)
    return environment.create_gym_environment(
        condition=environment.conditions[0],
        render_settings=RenderSettings(size=size, supersample_factor=2),
    )


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from pathlib import Path

import tyro
from imageio import v2 as imageio

from scripts.environments.registry import EnvironmentName, create_environment
from scripts.environments.render_settings import RenderSettings
from scripts.shared.log import create_logger
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass(frozen=True)
class Config:
    """Render one still image per task at its initial state, without the video HUD."""

    output_directory: Path = OUTPUT_DIRECTORY_PATH / "task_images"
    seed: int = 30_000
    size: int = 1_000
    environments: tuple[EnvironmentName, ...] = tuple(EnvironmentName)


def main() -> None:
    config = tyro.cli(f=Config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    for name in config.environments:
        environment = create_environment(name=name)
        gym_environment = environment.create_gym_environment(
            condition=environment.conditions[0],
            render_settings=RenderSettings(size=config.size, supersample_factor=2),
        )
        try:
            gym_environment.reset(seed=config.seed)
            frame = gym_environment.render()
        finally:
            gym_environment.close()
        path = config.output_directory / f"{name.value}.png"
        imageio.imwrite(uri=path, im=frame)
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()

import argparse
import json
import logging
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import zarr
from rich.pretty import pretty_repr


def find_project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in [path, *path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Project root not found")


PROJECT_ROOT = find_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output"


def load_push_t_zarr():
    zarr_file_path = PROJECT_ROOT / "expert_data" / "pusht_cchi_v7_replay.zarr"
    store = zarr.open(store=zarr_file_path, mode="r")

    # this removes the "does not support indexing" type checker warning
    # zarr.open() return type is too broad to support indexing
    # mode="r" with a path always returns a Group
    assert isinstance(store, zarr.Group)

    return (
        store["data/state"][:],
        store["data/action"][:],
        store["meta/episode_ends"][:],
    )


def create_logger():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)


def get_formatted_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_config_file_path():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("config", type=Path, help="Path to config file")
    arguments = argument_parser.parse_args()
    config_file_path: Path = arguments.config
    return config_file_path


def initialize_run[T](
    config_class: type[T], runs_output_directory_path: Path, logger: logging.Logger
) -> tuple[T, torch.device, Path]:
    config_file_path = get_config_file_path()

    with config_file_path.open() as config_file:
        config = config_class(**(json.load(config_file)))

    logger.info(pretty_repr(config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using torch device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU Memory: %.2f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    set_seed(seed=config.seed)

    run_output_directory_path = runs_output_directory_path / get_formatted_timestamp()
    run_output_directory_path.mkdir(parents=True)
    shutil.copy(src=config_file_path, dst=run_output_directory_path / "config.json")
    return config, device, run_output_directory_path

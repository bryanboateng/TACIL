import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import zarr


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

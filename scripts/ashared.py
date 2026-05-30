import hashlib
import json
import logging
import random
from typing import Any

import numpy as np
import torch
import zarr
from numpy import typing as npt
from zarr import Array, Group

from scripts.paths import PROJECT_ROOT


def load_push_t_zarr() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int64]
]:
    zarr_file_path = PROJECT_ROOT / "expert_data" / "pusht_cchi_v7_replay.zarr"
    store = zarr.open(store=zarr_file_path, mode="r")
    assert isinstance(store, Group)

    def _get(key: str, dtype: type) -> np.ndarray:
        group_member = store[key]
        assert isinstance(group_member, Array), f"{key!r} is not an Array"
        assert group_member.dtype == dtype, (
            f"{key!r} has dtype {group_member.dtype.name}, expected {dtype.__name__}"
        )
        data = group_member[:]
        assert isinstance(data, np.ndarray)
        return data

    states = _get(key="data/state", dtype=np.float32)
    actions = _get(key="data/action", dtype=np.float32)
    episode_ends = _get(key="meta/episode_ends", dtype=np.int64)
    return states, actions, episode_ends


def create_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)


def set_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_device_info(device: torch.device, logger: logging.Logger) -> None:
    logger.info("Using torch device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU Memory: %.2f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )


def compute_config_hash(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(serialized.encode()).hexdigest()[:8]

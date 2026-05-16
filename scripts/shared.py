import logging
import random
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import zarr
from zarr import Array, Group


def find_project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in [path, *path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Project root not found")


PROJECT_ROOT = find_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output"


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


def create_logger():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import hashlib
import json
import logging
import random
from typing import Any

import numpy as np
import torch


def create_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)


def set_seed(*, seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_device_info(*, device: torch.device, logger: logging.Logger) -> None:
    logger.info("Using torch device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU Memory: %.2f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )


def compute_config_hash(*, config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(serialized.encode()).hexdigest()[:8]

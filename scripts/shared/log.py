import logging

import torch


def create_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(name=__name__)


def log_device_info(*, device: torch.device, logger: logging.Logger) -> None:
    logger.info("Using torch device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(device=device))
        logger.info(
            "GPU Memory: %.2f GiB",
            torch.cuda.get_device_properties(device=device).total_memory / 2**30,
        )

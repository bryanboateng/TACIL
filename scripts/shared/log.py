import logging
import re
import subprocess

import torch


def create_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(name=__name__)


def log_device_info(*, device: torch.device, logger: logging.Logger) -> None:
    logger.info("Using torch device: %s", device)
    if device.type == "cuda":
        device_properties = torch.cuda.get_device_properties(device=device)
        logger.info(
            "GPU index: %s", _get_device_index(device_properties=device_properties)
        )
        logger.info("GPU name: %s", torch.cuda.get_device_name(device=device))
        logger.info(
            "GPU total memory: %.2f GiB", device_properties.total_memory / 2**30
        )


# ---- Private --------------------------------------------


def _get_device_index(*, device_properties: torch._C._CudaDeviceProperties) -> str:
    uuid = str(device_properties.uuid)

    try:
        listing = subprocess.run(
            args=["nvidia-smi", "-L"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unknown (nvidia-smi unavailable: {error})"

    match = re.search(
        pattern=rf"GPU (\d+): .*\(UUID: GPU-{re.escape(pattern=uuid)}\)", string=listing
    )
    if match is None:
        return f"unknown (no nvidia-smi match for {uuid})"
    return match.group(1)

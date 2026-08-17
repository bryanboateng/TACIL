import random

import numpy as np
import torch


def set_seed(*, seed) -> None:
    random.seed(a=seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)

    torch.use_deterministic_algorithms(mode=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

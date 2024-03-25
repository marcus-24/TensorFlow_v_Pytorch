import torch
from typing import Tuple

def get_workers() -> Tuple[str, int]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 0 if device == "cuda" else 2  # avoids runtime error since gpu cant have multiple workers
    print('device used: ', device)
    print('num workers: ', num_workers)
    torch.set_default_device(device)
    return device, num_workers
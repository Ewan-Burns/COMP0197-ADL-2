import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    """
    Set all random seeds to a fixed value and take a few random samples to ensure reproducibility.
    
    Args:
        seed (int): Seed value to use
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Print confirmation message for debugging
    print(f"Random seed set to {seed}")
    
    # Take a few random calls to make sure everything is properly seeded
    print(f"Random check: {random.random()}")
    print(f"Numpy check: {np.random.rand(1)[0]}")
    print(f"Torch check: {torch.rand(1)[0].item()}")

def worker_init_fn(worker_id):
    """
    Worker initialization function to ensure reproducible data loading.
    
    Args:
        worker_id (int): Worker ID from DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
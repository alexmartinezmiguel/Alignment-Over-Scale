"""
Miscellaneous utility functions (logging, seeding, data helpers).
"""

import os
import random
import logging
import argparse

import numpy as np
import torch


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    """Configure a basic console logger and return the module-level logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# DICTIONARY HELPERS
# =============================================================================

def split_dict(input_dict: dict, split_ratio: float = 0.8):
    """Randomly split a dictionary into two disjoint subsets.

    Args:
        input_dict: Dictionary to split.
        split_ratio: Fraction of keys assigned to the first subset.

    Returns:
        Tuple of two dictionaries.
    """
    keys = list(input_dict.keys())
    random.shuffle(keys)
    split_index = int(len(keys) * split_ratio)
    keys_a, keys_b = keys[:split_index], keys[split_index:]
    return {k: input_dict[k] for k in keys_a}, {k: input_dict[k] for k in keys_b}


def sample_dict(dictionary: dict, n: int) -> dict:
    """Sample *n* random key-value pairs from a dictionary.

    Args:
        dictionary: Source dictionary.
        n: Number of pairs to sample.

    Returns:
        Dictionary with at most *n* sampled pairs.
    """
    if n >= len(dictionary):
        return dictionary.copy()
    sampled_keys = random.sample(list(dictionary.keys()), n)
    return {key: dictionary[key] for key in sampled_keys}


def sample_n_samples(dictionary: dict, n: int) -> np.ndarray:
    """Sample *n* keys (without replacement) from a dictionary.

    Args:
        dictionary: Source dictionary.
        n: Number of keys to sample.

    Returns:
        NumPy array of sampled keys.
    """
    return np.random.choice(list(dictionary.keys()), size=n, replace=False)


# =============================================================================
# ARGUMENT PARSING HELPERS
# =============================================================================

def str2bool(v: str) -> bool:
    """Convert a string to a boolean value (for argparse compatibility).

    Accepts 'yes'/'true'/'t'/'y'/'1' as True and
    'no'/'false'/'f'/'n'/'0' as False.

    Args:
        v: String to convert.

    Returns:
        Boolean value.

    Raises:
        argparse.ArgumentTypeError: If the string is not recognised.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import os
import pickle

import pandas as pd

from .constants import LBD_PATH, V3_MODELS_DIR


# @functools.lru_cache(maxsize=None)
def load_total_ibd_bounds(
    lbd_path: str=LBD_PATH,
):
    df = pd.read_csv(lbd_path)
    df = df.fillna(0)
    return df


# @functools.lru_cache(maxsize=None)
def load_ibd_moments(
    min_seg_len: float=0,
    models_dir: str=V3_MODELS_DIR,
):
    """
    Load the moments of the conditional and unconditional IBD number and length distributions.
    These moments are for the marginal distribution of n and the marginal distribution of the
    total length L separately.

    Args:
        min_seg_len (int): The minimum segment length used to generate the moments.

    Returns:
        cond_dict (dict): A dictionary of the conditional moments.
            {m: {a: {mean_num_1: val, mean_L_1: val, std_L_1: val, mean_num_2: val, mean_L_2: val, std_L_2: val}}}
            where
                m: number of meioses
                a: number of common ancestors
                mean_num_1: mean number of IBD1 segments
                mean_L_1: mean total length of IBD1 segments
                std_L_1: standard deviation of total length of IBD1 segments
                mean_num_2: mean number of IBD2 segments
                mean_L_2: mean total length of IBD2 segments
                std_L_2: standard deviation of total length of IBD2 segments
        uncond_dict (dict): A dictionary of the unconditional moments. Same format as cond_dict.
    """  # noqa: E501
    cond_fp = os.path.join(
        models_dir,
        "ibd_moments",
        f"min_seg_len_{min_seg_len}",
        "cond_moments.pkl",
    )
    uncond_fp = os.path.join(
        models_dir,
        "ibd_moments",
        f"min_seg_len_{min_seg_len}",
        "uncond_moments.pkl",
    )
    cond_dict = pickle.loads(open(cond_fp, "rb").read())
    uncond_dict = pickle.loads(open(uncond_fp, "rb").read())
    return cond_dict, uncond_dict

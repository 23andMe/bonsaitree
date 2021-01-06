import ast
from dataclasses import dataclass
import json
import os
import functools
from typing import Any, Dict, List, Tuple

import numpy as np


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DISTN_CACHE_PATH = os.path.join(MODELS_DIR, "distn_dict.json")
AGE_DIFF_DISTN_PATH = os.path.join(MODELS_DIR, "age_diff_moments.json")


AUTO_GENOME_LENGTH = 3544.57
FULL_GENOME_LENGTH = 3725.41
MIN_SEG_LEN = 5  # Minimum length of an observed IBD segment (in cM) from whatever IBD inference tool we're using.
MIN_PARENT_CHILD_AGE_DIFF = 10
SMALL_LOG_PROB = -1e10

@dataclass(frozen=True)
class Distributions:
    dstn_dict: Dict[Tuple[int, ...], Dict[str, Any]]
    age_diff_moments: Dict[Tuple[int, ...], List[float]]
    genome_length: float = AUTO_GENOME_LENGTH
    min_seg_len: int = MIN_SEG_LEN

    def get_distribution(self, relation_tuple: Tuple[int, int, int]):
        """
        Retrieves precomputed distributions by relationship
        """
        distribution = self.dstn_dict.get(relation_tuple)
        if not distribution:
            # Attempt to retrieve relationship with up/down inverted
            distribution = self.dstn_dict.get(
                (relation_tuple[1], relation_tuple[0], relation_tuple[2])
            )

        if not distribution:
            # Attempt to retrieve distribution with up/down sum
            distribution = self.dstn_dict.get(
                (relation_tuple[0] + relation_tuple[1], relation_tuple[2])
            )

        if not distribution:
            raise KeyError(
                "Missing distribution for the following relationship: {}".format(
                    relation_tuple
                )
            )

        k_mean_1, k_std_1 = distribution["count"][1]
        T_mean_1, T_std_1 = distribution["total_len"][1]

        k_mean_2, k_std_2 = distribution["count"][2]
        T_mean_2, T_std_2 = distribution["total_len"][2]

        return (
            k_mean_1,
            k_std_1,
            T_mean_1,
            T_std_1,
            k_mean_2,
            k_std_2,
            T_mean_2,
            T_std_2,
        )

    def get_analytical_distant_relative_distribution(
        self, relation_tuple: Tuple[int, int, int]
    ):
        """
        Get the paramters for the log likelihood using the Huff et al (2011) formula. 
        (Equations 7, 8, and the paragraph before Eqn 7). This is for the case when
        two people are related by more than 14 meioses. We do not simulate IBD
        distributions for relationships this distant so we need to use the analytical
        distribution.
        """
        up_meioses = relation_tuple[0]
        down_meioses = relation_tuple[1]
        num_ancestors = relation_tuple[2]

        num_recs_per_gen = self.genome_length / 100
        g = up_meioses + down_meioses
        expected_mean_k = (
            num_ancestors * (num_recs_per_gen * g + 22) * (1 / (2 ** (g - 1)))
        )
        p_obs = np.exp(
            -g * self.min_seg_len / 100
        )  # Probability that a segment is observed, given that the minimum IBD length is min_seg_len

        k_mean_1 = expected_mean_k * p_obs
        k_std_1 = np.sqrt(k_mean_1)

        threshold = self.min_seg_len
        T_mean_1 = (100 / g - threshold) * k_mean_1
        T_std_1 = (100 / g) * np.sqrt(2 * k_mean_1)  # From the law of total variance

        T_mean_2, T_std_2 = (0, 5)  # Parameters for background IBD2

        return (k_mean_1, k_std_1, T_mean_1, T_std_1, T_mean_2, T_std_2)

@functools.lru_cache(maxsize=None)
def load_distributions(ibd_threshold=None) -> Distributions:
    """
    Loads json distributions models and age diff moments
    """
    
    # Load distributions for a given threshold if available
    try:
        thresholded_distn_path = DISTN_CACHE_PATH.replace(
            ".json", "_threshold_" + str(ibd_threshold) + ".json"
        )
        relation_distributions = _load_json_distns(thresholded_distn_path)
    except Exception:
        relation_distributions = _load_json_distns(DISTN_CACHE_PATH)

    # Load age diff moments
    age_diff_moments = _load_json_distns(AGE_DIFF_DISTN_PATH)

    return Distributions(relation_distributions, age_diff_moments)


def _load_json_distns(path):
    return unstringify_keys(json.loads(open(path).read()))


def stringify_keys(obj):
    if isinstance(obj, dict):
        new_obj = dict()
        for key, val in obj.items():
            if isinstance(key, frozenset):
                new_key = "f" + str(tuple(key))
            else:
                new_key = str(key)
            new_obj[new_key] = stringify_keys(val)
        return new_obj
    else:
        return obj


def unstringify_keys(obj):
    if isinstance(obj, dict):
        new_obj = dict()
        for key, val in obj.items():
            if key[0] == "f":
                new_key_list = ast.literal_eval(key[1:])
                new_key = frozenset(new_key_list)
            else:
                try:
                    new_key = ast.literal_eval(key)
                except Exception:
                    if key.count("inf") == 2:
                        new_key = (float("inf"), float("inf"), None)
                    else:
                        new_key = key
            new_obj[new_key] = unstringify_keys(val)
        return new_obj
    else:
        return obj

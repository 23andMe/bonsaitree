"""
The point prediction module is used to estimate relationship likelihoods between two or more individuals.

# profile_information
profile_info = {
    123: {
        age: 30,
        sex: 'M'
    }...
}

# ibd summary dict
ibd_summaries = {
    FrozenSet[unique_id_1, unique_id_2]: {
        total_half: float
        total_full: float
        num_half: int
        max_seg_cm: float
    }...
}

dist = distributions.load_distributions(ibd_threshold=0)
ppg = construct_point_prediction_group(profile_info, ibd_summaries)

(
    pairwise_predicted_relationship_type,
    pairwise_likelihoods_by_type
) = point_predictions(ppg, dist)
"""
from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from typing import Any, DefaultDict, Dict, FrozenSet, Iterator, List, Tuple

import numpy as np
import scipy.stats

from .distributions import Distributions, AUTO_GENOME_LENGTH


_RELATIONSHIP_TYPES: List[Tuple[int, int, int]] = [
    (1, 1, 2),  # Sibling
    (0, 1, 1),  # Child
    (1, 0, 1),  # Parent
    (0, 2, 1),  # Grandchild
    (2, 0, 1),  # Grandparent
    (0, 3, 1),  # Great grandchild
    (3, 0, 1),  # Great grandparent
    (2, 1, 2),  # Pibling
    (1, 2, 2),  # Nibling
    (1, 1, 1),  # Half-sibling
    (2, 1, 1),  # Half-pibling
    (1, 2, 1),  # Half-nibling
    (2, 2, 2),  # First cousin
    (1, 3, 2),  # Great nibling
    (3, 1, 2),  # Great pibling
    (1, 3, 1),  # Great half-nibling
    (2, 2, 1),  # Half first cousin
    (3, 1, 1),  # Great half-pibling
    (4, 0, 1),  # Great, great, grandparent
    (0, 4, 1),  # Great, great, grandchild
    # Distant
    (float("inf"), float("inf"), None),  # type: ignore
]


# Add more distant types up to 15th degree
for num_meioses in range(5, 13):
    for up_m in range(num_meioses + 1):
        down_m = num_meioses - up_m
        for num_ancs in [1, 2]:
            if num_ancs == 2 and (up_m == 0 or down_m == 0):
                continue
            _RELATIONSHIP_TYPES.append((up_m, down_m, num_ancs))


# Freeze list of relationship types in a tuple
RELATIONSHIP_TYPES: Tuple[Tuple[int, int, int], ...] = tuple(_RELATIONSHIP_TYPES)


@dataclass(frozen=True)
class ProfileInformation:
    """
    Immutable dataclass for any profile information for a particular individual
    """

    __slots__ = ["unique_id", "age", "sex"]

    unique_id: int
    age: int
    sex: str


@dataclass(frozen=True)
class IBDSummary:
    """
    Immutable dataclass for segment data between two individuals
    """

    __slots__ = [
        "total_half_len",
        "total_full_len",
        "half_segment_count",
        "max_segment_cm",
    ]

    total_half_len: float
    total_full_len: float
    half_segment_count: int
    max_segment_cm: float


@dataclass(frozen=True)
class PointPredictionGroup:
    """
    Dataclass that represents a group of individuals that point prediction is being
    run on.
    """

    __slots__ = ["profile_infos", "pairwise_segment_summaries", "pairwise_list"]
    profile_infos: Dict[int, ProfileInformation]
    pairwise_segment_summaries: Dict[FrozenSet[int], IBDSummary]
    pairwise_list: List[Tuple[int, int]]

    @property
    def total_half_len_list(self) -> List[float]:
        return [
            self.pairwise_segment_summaries[frozenset(pair)].total_half_len
            for pair in self.pairwise_list
        ]

    @property
    def total_full_len_list(self) -> List[float]:
        return [
            self.pairwise_segment_summaries[frozenset(pair)].total_full_len
            for pair in self.pairwise_list
        ]

    @property
    def half_seg_count_list(self) -> List[int]:
        return [
            self.pairwise_segment_summaries[frozenset(pair)].half_segment_count
            for pair in self.pairwise_list
        ]

    @property
    def max_seg_len_list(self) -> List[float]:
        return [
            self.pairwise_segment_summaries[frozenset(pair)].max_segment_cm
            for pair in self.pairwise_list
        ]


def construct_point_prediction_group(
    profile_information: Dict[int, Dict[str, Any]],
    summaries: Dict[FrozenSet[int], Dict[str, Any]],
) -> PointPredictionGroup:
    """
    Args:
        profile_information: Any information attached to
            specific individuals that we care about.
        summaries: A dictionary of summary stats for each unique pair of individuals

        # profile_information
        {
            123: {
                age: 30,
                sex: 'M'
            }...
        }

        # ibd summary dict
        {
            FrozenSet[unique_id_1, unique_id_2]: {
                total_half: float
                total_full: float
                num_half: int
                max_seg_cm: float
            }...
        }
    """
    # Convert profile info dictionaries to frozen dataclass structures
    profile_infos = {
        unique_id: ProfileInformation(unique_id, profile["age"], profile["sex"])
        for unique_id, profile in profile_information.items()
    }

    unique_ids_set = {*profile_infos}

    pairwise_segment_summaries: Dict[FrozenSet[int], IBDSummary] = {}
    pairwise_list: List[Tuple[int, int]] = []

    # Typeshed stub for permutations returns Iterator[Tuple, [Any, ...]]
    # even though repeat is specified.
    id_permutations: Iterator[Any] = permutations(unique_ids_set, r=2)
    for id_pair in id_permutations:
        frozen_set_key = frozenset(id_pair)
        pairwise_segment_summaries[frozen_set_key] = IBDSummary(
            summaries[frozen_set_key]["total_half"],
            summaries[frozen_set_key]["total_full"],
            summaries[frozen_set_key]["num_half"],
            summaries[frozen_set_key]["max_seg_cm"],
        )
        pairwise_list.append(id_pair)

    # Initialize a PointPredictionGroup dataclass for a group of individuals.
    return PointPredictionGroup(
        profile_infos, pairwise_segment_summaries, pairwise_list,
    )


def point_predictions(
    point_prediction_group: PointPredictionGroup,
    distribution_models: Distributions,
    renormalize_age_likes: bool = False,
) -> Tuple[
    Dict[int, Dict[int, Tuple[int, int, int]]],
    Dict[int, Dict[int, Dict[Tuple[int, int, int], float]]],
]:
    """
    Args:
        point_prediction_group: A frozen dataclass containing all bio information and
            segment summary data
        distribution_models: A frozen dataclass containing precomputed distributions
            by relationship and age difference moments
        renormalize_age_likes: Makes all likelihoods on the same order of magnitude
    """
    # Estimates relationships using scipy for a point prediction group
    likelihoods = scipy_estimate_relationships(
        point_prediction_group, distribution_models,
    )

    # Maximizes likelihoods retrieved from relationship estimation
    estimated_relationships = maximize_likelihoods(likelihoods)

    # Constructs a dictionary mapping each pair of individuals
    # with shared IBD segment data to a predicted relationship type
    # and a list of likelihoods for each relationship type.
    pairwise_predicted_relationship_type: DefaultDict[
        int, Dict[int, Tuple[int, int, int]]
    ] = defaultdict(dict)
    pairwise_likelihoods_by_type: DefaultDict[
        int, Dict[int, Dict[Tuple[int, int, int], float]]
    ] = defaultdict(dict)

    for idx, pairwise_relation in enumerate(point_prediction_group.pairwise_list):
        id_1, id_2 = pairwise_relation
        pairwise_predicted_relationship_type[id_1][id_2] = estimated_relationships[idx]

        likelihoods_list = {key: val[idx] for key, val in likelihoods.items()}
        pairwise_likelihoods_by_type[id_1][id_2] = likelihoods_list

    # Override default factory to remove defaultdict behavior
    pairwise_predicted_relationship_type.default_factory = None
    pairwise_likelihoods_by_type.default_factory = None

    return (
        pairwise_predicted_relationship_type,
        pairwise_likelihoods_by_type,
    )


def scipy_estimate_relationships(
    point_prediction_group: PointPredictionGroup,
    models: Distributions,
    renormalize_age_likes: bool = False,
    target_relationship_types=RELATIONSHIP_TYPES,
) -> Dict[Tuple[int, int, int], List[float]]:
    """
    Args:
        point_prediction_group: A frozen dataclass containing all bio information and
            segment summary data
        distribution_models: A frozen dataclass containing precomputed distributions
            by relationship and age difference moments
        renormalize_age_likes: Makes all likelihoods on the same order of magnitude
        target_relationship_types: Target relationships to compute estimates for
    """
    # Cast point_prediction_group lists to numpy arrays so that they
    # can be passed into the scipy module.
    total_half_len_list = np.array(point_prediction_group.total_half_len_list)
    total_full_len_list = np.array(point_prediction_group.total_full_len_list)
    half_seg_count_list = np.array(point_prediction_group.half_seg_count_list)

    # Errors loudly if we see ibd_segment summaries for people without bios
    age1_list, age2_list = (
        np.array(
            [
                point_prediction_group.profile_infos[individual].age
                for individual in rel_list
            ]
        )
        for rel_list in zip(*point_prediction_group.pairwise_list)
    )

    likelihoods = {}
    for relation_type in target_relationship_types:
        (
            k_mean_1,
            k_std_1,
            T_mean_1,
            T_std_1,
            k_mean_2,
            k_std_2,
            T_mean_2,
            T_std_2,
        ) = models.get_distribution(relation_type)

        # Log likelyhood based on half segment counts
        if relation_type == (1, 0, 1) or relation_type == (0, 1, 1):
            log_like = scipy.stats.expon.logpdf(
                half_seg_count_list, loc=k_mean_1, scale=k_std_1
            )
        else:
            log_like = scipy.stats.norm.logpdf(
                half_seg_count_list, loc=k_mean_1, scale=k_std_1
            )

        # Log likelyhood based on total half segments
        log_like += scipy.stats.norm.logpdf(
            total_half_len_list, loc=T_mean_1, scale=T_std_1
        )

        # Log likelyhood based on total full segments
        if relation_type == (1, 1, 2):
            log_like += scipy.stats.norm.logpdf(
                total_full_len_list, loc=T_mean_2, scale=T_std_2
            )
        else:
            log_like += scipy.stats.expon.logpdf(
                total_full_len_list, loc=T_mean_2, scale=T_std_2
            )

        # Log likelyhood based on age
        log_like += log_likelihoods_by_age(
            age1_list,
            age2_list,
            relation_type,
            models.age_diff_moments,
            renormalize_age_likes=renormalize_age_likes,
        )
        likelihoods[relation_type] = log_like

    return likelihoods


def maximize_likelihoods(
    likelihoods: Dict[Tuple[int, int, int], List[float]]
) -> List[Tuple[int, int, int]]:
    """
    Args:
        likelihoods: A mapping of relationship types to list of likelihoods.
            Each index in the list of likelihoods corresponds to a particular
            pair of individual from a point_prediction_group.

    Returns
        A list of most likely relationship types
    """
    # Check that all lists of likelihoods have the same length for each relationship
    iterator = iter(likelihoods.values())
    length = len(next(iterator))
    if not all(len(l) == length for l in iterator):
        raise ValueError("Assymetrical likelyhood counts between relationships")

    maximal_likelihoods = [-float("inf")] * length
    estimated_relationships = [Tuple[int, int, int]] * length

    for relation_type, likelihoods_list in likelihoods.items():
        for idx, likelyhood in enumerate(likelihoods_list):
            if likelyhood > maximal_likelihoods[idx]:
                maximal_likelihoods[idx] = likelyhood
                estimated_relationships[idx] = relation_type

    return estimated_relationships


def log_likelihoods_by_age(
    age1_list: List[int],
    age2_list: List[int],
    relation_type: Tuple[int, int, int],
    age_diff_moments: Dict[Tuple[int, ...], List[float]],
    renormalize_age_likes: bool = False,
) -> List[float]:
    """
    Retrieve likelihoods for a relationship for list of individuals

    Args:
        age1_list: List of individuals representing half of a pair
        age2_list: List of individuals representing the other half of a pair
        relation_type: Relationship tuple to evalute for
        age_diff_moments: Precomputed distributions to use
        renormalize_age_likes: Makes all likelihoods on the same order of magnitude
    """
    # If unrelated, don't alter likelihood. In the future, we
    # might want to replace this with the likelihood for a random unrelated pair
    if not relation_type[2]:
        return [0]

    diff_mean, diff_std = get_age_diff_mean_and_std(relation_type, age_diff_moments)

    # # Compute the age difference for each pair of individuals, filling in with the diff_mean
    age_diffs = np.array(
        [
            elt1 - elt2 if (elt1 is not None) and (elt2 is not None) else diff_mean
            for elt1, elt2 in zip(age1_list, age2_list)
        ]
    )

    age_log_like = scipy.stats.norm.logpdf(age_diffs, loc=diff_mean, scale=diff_std)

    if renormalize_age_likes:
        # Renormalize likelihoods by comparing to the reference likelihood. Do this by getting
        # the relationship degree with the same age diference and number of common ancestors,
        # but the smallest total number of meioses.

        min_deg = min(relation_type[:2])
        equiv_rel_deg_0 = relation_type[0] - min_deg + 1
        equiv_rel_deg_1 = relation_type[1] - min_deg + 1
        equiv_rel_deg = (equiv_rel_deg_0, equiv_rel_deg_1, relation_type[2])
        equiv_diff_mean, equiv_diff_std = get_age_diff_mean_and_std(
            equiv_rel_deg, age_diff_moments
        )

        log_peak_height = scipy.stats.norm.logpdf(
            diff_mean, loc=diff_mean, scale=diff_std
        )
        log_ref_peak_height = scipy.stats.norm.logpdf(
            equiv_diff_mean, loc=equiv_diff_mean, scale=equiv_diff_std
        )

        renorm_constant = log_ref_peak_height - log_peak_height

        age_log_like += renorm_constant

    return age_log_like


def get_age_diff_mean_and_std(
    relation_type: Tuple[int, int, int],
    age_diff_moments: Dict[Tuple[int, ...], List[float]],
) -> Tuple[float, float]:
    """
    Get the mean and variance of the age difference distribution for degree rel_deg.
    If we have empirically determined these values, get them from the list. Otherwise,
    estimate them by adding means and variances of age difference distributions.
    """
    # If we previously fit a distribution from self-reported data
    relation_str = relation_type
    if relation_str in age_diff_moments:
        diff_mean = age_diff_moments[relation_str][0]
        diff_std = age_diff_moments[relation_str][1]

    else:
        # Make a guess about the distribution from self-reported data by adding means and variances for
        # parent-child relationships, sibships, or half-sibships
        num_up_meioses = relation_type[0]
        num_down_meioses = relation_type[1]
        num_ancestors = relation_type[2]

        if (
            num_ancestors is None
        ):  # Then we are considering the (inf,inf,None) unrelated type.
            num_up_meioses = 3  # Give them the same age distribution as half second cousins. I.e., most likely of the same age, but with a big variance
            num_down_meioses = 3
            num_ancestors = 1

        if num_up_meioses == 0:
            diff_mean = (
                num_down_meioses * age_diff_moments[(0, 1, 1)][0]
            )  # Mean is num_down_meioses times the mean for child
            diff_std = np.sqrt(
                num_down_meioses * (age_diff_moments[(0, 1, 1)][1] ** 2)
            )  # Var is num_down_meioses times the variance for child
        elif num_down_meioses == 0:
            diff_mean = (
                num_up_meioses * age_diff_moments[(1, 0, 1)][0]
            )  # Mean is num_up_meioses times the mean for parent
            diff_std = np.sqrt(
                num_up_meioses * (age_diff_moments[(1, 0, 1)][1] ** 2)
            )  # Var is num_up_meioses times the variance for parent
        else:
            diff_mean = (num_up_meioses - 1) * age_diff_moments[(1, 0, 1)][0]
            diff_mean += (num_down_meioses - 1) * age_diff_moments[(0, 1, 1)][0]
            diff_mean += age_diff_moments[(1, 1, 2)][0]
            diff_var = (num_up_meioses - 1) * (age_diff_moments[(1, 0, 1)][1] ** 2)
            diff_var += (num_down_meioses - 1) * (age_diff_moments[(0, 1, 1)][1] ** 2)

            if num_ancestors == 1:
                diff_var += age_diff_moments[(1, 1, 1)][1] ** 2
            elif num_ancestors == 2:
                diff_var += age_diff_moments[(1, 1, 2)][1] ** 2

            diff_std = np.sqrt(diff_var)

    return diff_mean, diff_std


def get_distant_rel_log_like_by_ids(
    unique_id_1: int,
    unique_id_2: int,
    up_meioses: int,
    down_meioses: int,
    num_ancestors: int,
    point_prediction_group: PointPredictionGroup,
    distribution_models: Distributions,
) -> float:
    """
    Predicts likelyhood of a specific distant relationship between two individuals

    Args:
        unique_id_1: Id of first individual
        unique_id_2: Id of second individual
        point_prediction_group: Struct containing ibd_segment summary and bio_info data for id_1 and id_2
        distribution_models: Struct with preloaded distributions
        up_meioses: Degrees up to common ancestor
        down_meioses: Degrees down from common ancestor
        num_ancestors: Numer of shared ancestors
    """
    segment_summary = point_prediction_group.pairwise_segment_summaries.get(
        frozenset((unique_id_1, unique_id_2))
    )
    total_half_len = segment_summary.total_half_len if segment_summary else 0
    total_full_len = segment_summary.total_full_len if segment_summary else 0
    half_segment_count = segment_summary.half_segment_count if segment_summary else 0
    age1 = point_prediction_group.profile_infos[unique_id_1].age
    age2 = point_prediction_group.profile_infos[unique_id_2].age

    relation_type = (up_meioses, down_meioses, num_ancestors)

    try:
        (
            k_mean_1,
            k_std_1,
            T_mean_1,
            T_std_1,
            _,
            _,
            T_mean_2,
            T_std_2,
        ) = distribution_models.get_distribution(relation_type)
    except Exception:
        (
            k_mean_1,
            k_std_1,
            T_mean_1,
            T_std_1,
            T_mean_2,
            T_std_2,
        ) = distribution_models.get_analytical_distant_relative_distribution(
            relation_type
        )

    log_like = scipy.stats.norm.logpdf(half_segment_count, loc=k_mean_1, scale=k_std_1)
    log_like += scipy.stats.norm.logpdf(total_half_len, loc=T_mean_1, scale=T_std_1)
    log_like += scipy.stats.expon.logpdf(total_full_len, loc=T_mean_2, scale=T_std_2)

    age_log_like = log_likelihoods_by_age(
        [age1],
        [age2],
        (up_meioses, down_meioses, num_ancestors),
        distribution_models.age_diff_moments,
    )
    log_like += age_log_like

    return log_like


def is_twin_pair(
    total_half_len: float = None,
    total_full_len: float = None,
    sex1: str = None,
    sex2: str = None,
):
 """
 Checks if two individuals are twins
 is involved with filtering out twins prior to computes
 """
 if total_half_len is None:
     return False
 if total_full_len is None:
     return False

 if total_half_len < 0.95 * AUTO_GENOME_LENGTH:
     return False
 elif total_full_len < 0.95 * AUTO_GENOME_LENGTH:
     return False
 elif sex1 != sex2:
     return False
 else:
     return True

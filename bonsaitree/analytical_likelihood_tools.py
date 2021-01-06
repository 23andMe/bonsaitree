from typing import Any, Dict, List, Set, Tuple, FrozenSet, Optional

import copy
import numpy as np
import scipy.stats

try:
    from scipy.special import logsumexp
except:
    from scipy.misc import logsumexp
from itertools import combinations

from .node_dict_tools import get_root_to_desc_degrees, get_desc_deg_dict
from .distributions import load_distributions, AUTO_GENOME_LENGTH


INF = float("inf")
GENOME_LENGTH = AUTO_GENOME_LENGTH


def get_ibd_pattern_log_prob(
    node: int,
    node_dict: Dict[int, Dict[int, int]],
    ibd_presence_absence_dict: Dict[int, bool],
) -> Tuple[float, float]:
    """
    Compute a tuple, (n0,n1), where n0 is the probability of observing the
    haplotype presence-absence pattern in the leaves descended from 'node'
    if the haplotype was not transmitted as far as node and n1 is the probability
    of observing the haplotype presence-absence pattern in the leaves descended
    from node if the haplotype was transmitted as far as node.
    Args:
        node_dict : a dict of the form { node : {desc1 : deg1, desc2 : deg2, ....} }
                    node_dict skips omits nodes that are ancestral to only one person.
        ibd_presence_absence: dictionary of the form {leaf_i : o_i, ...}, where o_i == True
                   if the haplotype is observed in leaf_i and o_i == False, otherwise. 
                   Typically the keys of ibd_presence_absence_dict are leaves, but they
                   can be any independent set of nodes, which, if the tree were truncated
                   at them would comprise the full leaf set at that level of the tree.
        node : id of the node whose descendants correspond to the keys in ibd_presence_absence_dict.
    """

    # If the node is a leaf, or if the node is its own parent (so it's an orphan)
    if (node not in node_dict) or ((node in node_dict[node]) and (len(node_dict[node]) == 1)):
        state = ibd_presence_absence_dict.get(node, False)
        if state:
            return (-INF, 0)
        else:
            return (0, -INF)

    desc_node_dict = node_dict[node]
    n0 = 0.0
    n1 = 0.0
    for desc_node, g in desc_node_dict.items():
        desc_log_prob = get_ibd_pattern_log_prob(desc_node, node_dict, ibd_presence_absence_dict)
        if g > 0:
            log_trans_prob = np.log(1 - 2 ** (-g))
        else:
            log_trans_prob = -INF
        n0 += desc_log_prob[0]
        n1 += logsumexp(
            [desc_log_prob[0] + log_trans_prob, desc_log_prob[1] - g * np.log(2)]
        )
    return (n0, n1)


def get_log_prob_ibd(
    node_dict: Dict[int, Dict[int, int]],
    root_id: int,
    left_common_anc: int,
    right_common_anc: int,
    num_common_ancs: int,
) -> float:
    """
    Compute the log probability that an IBD segment is observed at a locus where
    the ibd segment arose in root_id and is observed between at least one descendant
    of left_common_anc and at least one descendant of right_common_anc.
    Args:
        node_dict : a dict of the form { node : {desc1 : deg1, desc2 : deg2, ....} }
                    node_dict skips omits nodes that are ancestral to only one person.
        root_id : ID in which the IBD segment arose.
        left_common_anc: common ancestor of a clade.
        right_common_anc: common ancestor of a clade.
        num_common_ancs: number of common ancestors (including root_id and possibly a spouse)
                         in which IBD semgments observed in the descendants of left_common_anc
                         and right_common_anc may have arisen.
    """
    left_num_generations = node_dict[root_id][left_common_anc]
    right_num_generations = node_dict[root_id][
        right_common_anc
    ]  # Number of generations between the root (founder) and the right common ancestor.

    if left_common_anc in node_dict:
        left_absence_dict = {desc: False for desc in node_dict[left_common_anc]}
        left_log_prob_tuple = get_ibd_pattern_log_prob(
            node=left_common_anc,
            node_dict=node_dict,
            ibd_presence_absence_dict=left_absence_dict,
        )  # Log probability that the segment is not passed on to any left descendant.
    else:  # left_common_anc is a leaf node. So the prob we dont observe it given that it was not passed to itself is 1.
        left_log_prob_tuple = (0, -INF)

    if right_common_anc in node_dict:
        right_absence_dict = {desc: False for desc in node_dict[right_common_anc]}
        right_log_prob_tuple = get_ibd_pattern_log_prob(
            node=right_common_anc,
            node_dict=node_dict,
            ibd_presence_absence_dict=right_absence_dict,
        )  # Log probability that the segment is not passed on to any right descendant.
    else:  # right_common_anc is a leaf node. So the prob we dont observe it given that it was not passed to itself is 1.
        right_log_prob_tuple = (0, -INF)

    left_log_prob_anc = -left_num_generations * np.log(2)  # Pr(Segment passedto the left common ancestor)
    left_log_prob_not_anc = logsumexp([0, left_log_prob_anc], b=[1, -1])  # Pr(Segment not passed to the left common ancestor)
    right_log_prob_anc = -right_num_generations * np.log(2)  # Pr(Segment passed on to the right common ancestor)
    right_log_prob_not_anc = logsumexp([0, right_log_prob_anc], b=[1, -1])  # Pr(Segment not passed to the right common ancestor).

    left_log_prob_0 = (left_log_prob_tuple[0] + left_log_prob_not_anc)  # Pr(data | left_anc = False) Pr(left_anc = False)
    left_log_prob_1 = (left_log_prob_tuple[1] + left_log_prob_anc)  # Pr(data | left_anc = True) Pr(left_anc = True)
    right_log_prob_0 = right_log_prob_tuple[0] + right_log_prob_not_anc
    right_log_prob_1 = right_log_prob_tuple[1] + right_log_prob_anc

    left_log_prob = logsumexp([left_log_prob_0, left_log_prob_1])  # Pr(Segment not passed to any left leaf)
    right_log_prob = logsumexp([right_log_prob_0, right_log_prob_1])

    log_prob_no_ibd_1_allele = logsumexp(
        [left_log_prob, right_log_prob, left_log_prob + right_log_prob], b=[1, 1, -1]
    )  # Pr(allele is not seen IBD among two clades)
    log_prob_no_ibd = (
        2 * num_common_ancs * log_prob_no_ibd_1_allele
    )  # Pr(No IBD due to any of the 2 * num_ancs alleles from the ancestors)

    return logsumexp([0, log_prob_no_ibd], b=[1, -1])  # Pr(IBD due to some allele is seen among the two clades)


def get_lambda_list(
    node_dict: Dict[int, Dict[int, int]],
    indep_leaf_set1: Set,
    indep_leaf_set2: Set,
    root_id: int,
    left_common_anc: int,
    right_common_anc: int,
    num_common_ancs: int,
) -> List[float]:
    """
    Get parameters (1/mean) of the exponential distributions 
    describing the lengths of segments shared between the members 
    of indep_leaf_set1 and indep_leaf_set2.
    Args:
        node_dict: a dict of the form { node : {desc1 : deg1, desc2 : deg2, ....} }
                node_dict skips omits nodes that are ancestral to only one person.
        indep_leaf_set1: set of leaves in one clade
        indep_leaf_set2: set of leaves in the other clade
        root_id : common ancestor of indep_leaf_set1 and indep_leaf_set2
        left_common_anc : common ancestor of indep_leaf_set1
        right_common_anc : common ancestor of indep_leaf_set2
        num_common_ancs: number of common ancestors (including root_id and maybe 
                         their spouse) of indep_leaf_set1 and indep_leaf_set2
    """
    left_anc_to_desc_deg_dict = get_desc_deg_dict(left_common_anc, node_dict)
    right_anc_to_desc_deg_dict = get_desc_deg_dict(right_common_anc, node_dict)

    left_anc_to_desc_deg_dict = {i : left_anc_to_desc_deg_dict[i] for i in indep_leaf_set1}
    right_anc_to_desc_deg_dict = {i : right_anc_to_desc_deg_dict[i] for i in indep_leaf_set2}

    root_to_left_anc_deg = node_dict[root_id][left_common_anc]
    root_to_right_anc_deg = node_dict[root_id][right_common_anc]
    pairwise_deg_list = [
        root_to_left_anc_deg
        + root_to_right_anc_deg
        + left_deg
        + right_deg
        + 1
        - num_common_ancs
        for left_deg in left_anc_to_desc_deg_dict.values()
        for right_deg in right_anc_to_desc_deg_dict.values()
    ]
    return [deg / 100 for deg in pairwise_deg_list]


def get_expected_seg_length_and_squared_length_for_leaf_subset(
    node_dict: Dict[int, Dict[int, int]],
    indep_leaf_set1: Set,
    indep_leaf_set2: Set,
    root_id: int,
    left_common_anc: int,
    right_common_anc: int,
    num_common_ancs: int,
) -> Tuple[float, float]:
    """
    Get the expected length of a segment shared IBD between clade1 and 
    clade2 where clade1 is made up pof indep_leaf_set1 and clade2 is 
    made up of indep_leaf_set2.
    Args:
        node_dict: a dict of the form { node : {desc1 : deg1, desc2 : deg2, ....} }
                node_dict skips omits nodes that are ancestral to only one person.
        indep_leaf_set1: set of leaves in one clade
        indep_leaf_set2: set of leaves in the other clade
        root_id : common ancestor of indep_leaf_set1 and indep_leaf_set2
        left_common_anc : common ancestor of indep_leaf_set1
        right_common_anc : common ancestor of indep_leaf_set2
        num_common_ancs: number of common ancestors (including root_id and 
                         maybe their spouse) of indep_leaf_set1 and indep_leaf_set2
    """
    lambda_list = get_lambda_list(
        node_dict,
        indep_leaf_set1,
        indep_leaf_set2,
        root_id,
        left_common_anc,
        right_common_anc,
        num_common_ancs,
    )
    num_lambdas = len(lambda_list)
    expected_length = 0
    expected_squared_length = 0
    for rval in range(1, num_lambdas + 1):
        sgn = (-1) ** (rval + 1)
        for lambda_tuple in combinations(lambda_list, r=rval):
            lambda_sum = sum(lambda_tuple)
            expected_length += sgn / (lambda_sum)
            expected_squared_length += sgn / (lambda_sum ** 2)
    return (expected_length, expected_squared_length)


def get_var_total_length_approx(
    node_dict: Dict[int, Dict[int, int]],
    indep_leaf_set1: Set,
    indep_leaf_set2: Set,
    root_id: int,
    left_common_anc: int,
    right_common_anc: int,
    num_common_ancs: int,
) -> Tuple[float, float, float]:
    """
    Approximate the variance in the total length of IBD observed between two clades.
    Args:
        node_dict: a dict of the form { node : {desc1 : deg1, desc2 : deg2, ....} }
                node_dict skips omits nodes that are ancestral to only one person.
        indep_leaf_set1: set of leaves in one clade
        indep_leaf_set2: set of leaves in the other clade
        root_id : common ancestor of indep_leaf_set1 and indep_leaf_set2
        left_common_anc : common ancestor of indep_leaf_set1
        right_common_anc : common ancestor of indep_leaf_set2
        num_common_ancs: number of common ancestors (including 
                         root_id and maybe their spouse) of 
                         indep_leaf_set1 and indep_leaf_set2
    """

    log_prob_ibd = get_log_prob_ibd(
        node_dict, root_id, left_common_anc, right_common_anc, num_common_ancs
    )

    num_leaves1 = len(indep_leaf_set1)
    num_leaves2 = len(indep_leaf_set2)

    deg_dict = get_root_to_desc_degrees(root_id, node_dict)
    expected_pattern1 = {uid: 2 ** (-deg_dict[uid]) for uid in indep_leaf_set1}
    expected_pattern2 = {uid: 2 ** (-deg_dict[uid]) for uid in indep_leaf_set2}

    expected_num1 = sum(expected_pattern1.values())  # expected number of leaves with observed IBD
    expected_num2 = sum(expected_pattern2.values())  # expected number of leaves with observed IBD

    num_subset1 = max(1, round(expected_num1))  # choose number of leaves in indep_leaf_set1 with observed IBD
    num_subset2 = max(1, round(expected_num2))

    max_like_pattern: Dict[int, bool] = dict()
    max_log_like = -INF
    for leaf_tuple1 in combinations(indep_leaf_set1, r=num_subset1):
        for leaf_tuple2 in combinations(indep_leaf_set2, r=num_subset2):
            leaf_subset1 = set(leaf_tuple1)
            leaf_subset2 = set(leaf_tuple2)
            ibd_presence_absence_dict = {leaf_id: True for leaf_id in leaf_subset1}
            ibd_presence_absence_dict.update(
                {leaf_id: False for leaf_id in indep_leaf_set1 - leaf_subset1}
            )
            ibd_presence_absence_dict.update(
                {leaf_id: True for leaf_id in leaf_subset2}
            )
            ibd_presence_absence_dict.update(
                {leaf_id: False for leaf_id in indep_leaf_set2 - leaf_subset2}
            )
            ibd_pattern_log_prob = get_ibd_pattern_log_prob(
                root_id, node_dict, ibd_presence_absence_dict
            )[1]

            if ibd_pattern_log_prob > max_log_like:
                max_log_like = ibd_pattern_log_prob
                max_like_pattern = ibd_presence_absence_dict

    leaf_subset1 = {uid for uid in indep_leaf_set1 if max_like_pattern.get(uid)}
    leaf_subset2 = {uid for uid in indep_leaf_set2 if max_like_pattern.get(uid)}


    (
        expected_len,
        expected_squared_len,
    ) = get_expected_seg_length_and_squared_length_for_leaf_subset(
        node_dict=node_dict,
        indep_leaf_set1=leaf_subset1,
        indep_leaf_set2=leaf_subset2,
        root_id=root_id,
        left_common_anc=left_common_anc,
        right_common_anc=right_common_anc,
        num_common_ancs=num_common_ancs,
    )

    return (
        2 * np.exp(log_prob_ibd) * GENOME_LENGTH * expected_squared_len / expected_len,
        expected_len,
        expected_squared_len,
    )


def get_log_like_total_length_normal(
    L_tot: float,
    mean: float,
    var: float,
) -> float:
    """
    Get the log likelihood of observing the total merged length of IBD, assuming a normal distribution
    Args:
        L_tot: total amount of IBD (in cM) shared among two sets of individuals
        mean: Expected total length of IBD
        var: variance of total length of IBD
    """
    std = np.sqrt(var)
    return scipy.stats.norm.logpdf(L_tot, loc=mean, scale=std)


def get_background_test_pval_gamma(
    L_tot: float,
    mean: float,
    var: float,
) -> float:
    """
    Evaluate the CDF of the probability of the total length at the point L_tot
    Args:
        L_tot: total amount of IBD (in cM) shared among two sets of individuals
        mean: Expected total length of IBD
        var: variance of total length of IBD
    """
    k = (mean ** 2) / var
    theta = var / mean
    log_cdf = scipy.stats.gamma.logcdf(L_tot, k, scale=theta) # log[Pr(L < l)]
    log_surv = logsumexp([0, log_cdf], b=[1, -1]) # log[1 - Pr(L > l)]
    log_pval = min(log_cdf, log_surv) 
    return np.exp(log_pval)


def get_background_test_pval_normal(
    L_tot: float,
    mean: float,
    var: float,
    expected_count: float,
) -> float:
    """
    Evaluate the CDF of the probability of the total length at the point L_tot using a normal approximation.
    Args:
        L_tot: total amount of IBD (in cM) shared among two sets of individuals
        mean: Expected total length of IBD
        var: variance of total length of IBD
    """
    if L_tot > 0:
        log_like = get_log_like_total_length_normal(L_tot, mean, var) + logsumexp(
            [0, -expected_count], b=[1, -1]
        )
    else:
        log_like = -expected_count

    return np.exp(log_like)

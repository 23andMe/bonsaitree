from typing import Any, Dict, List, Set, Tuple, FrozenSet, Optional

import copy
import numpy as np
import scipy.stats
try:
    from scipy.special import logsumexp
except:
    from scipy.misc import logsumexp
from itertools import combinations, product

from .exceptions import UnlikelyRelationshipException
from .pedigree_object import connect_pedigrees_through_founders
from .exceptions import BuildFailException
from .analytical_likelihood_tools import get_log_prob_ibd, get_ibd_pattern_log_prob, get_background_test_pval_gamma, get_var_total_length_approx, get_log_like_total_length_normal, get_background_test_pval_normal
from .node_dict_tools import get_min_id, get_node_dict_founder_set, get_node_dict, get_node_dict_for_root, adjust_node_dict_to_common_anc, extract_down_node_dict, get_leaf_set
from .ibd_tools import check_overlap, get_segment_length_list, get_ibd_segs_between_sets, merge_ibd_segs, get_related_sets
from .point_predictor import log_likelihoods_by_age
from .distributions import load_distributions, AUTO_GENOME_LENGTH, MIN_PARENT_CHILD_AGE_DIFF

INF = float('inf')
UNREL_DEG = (INF,INF,None)
GENOME_LENGTH = AUTO_GENOME_LENGTH



def infer_degree_generalized_druid(
    leaf_set1 : Set[int],
    leaf_set2 : Set[int],
    node_dict1 : Dict[int, Dict[int, int]],
    node_dict2 : Dict[int, Dict[int, int]],
    L_merged_tot : int,
) -> int:
    """
    Use our generalized version of the DRUID estimator to infer the degree separating the 
    ancestors of node_dict1 and node_dict2
    Args:
        leaf_set1 : set of genotyped IDs in the first pedigree
        leaf_set2 : set of genotyped IDs in the second pedigree
        node_dict1: node dict corresponding to leaf_set1 : a dict of the form
                    { node : {desc1 : deg1, desc2 : deg2, ....} }
        node_dict2: node dict corresponding to leaf_set2 : a dict of the form
                    { node : {desc1 : deg1, desc2 : deg2, ....} }
        L_merged_tot : total amount of IBD shared between leaf_set1 and leaf_set2
    """

    if node_dict1:
        min_id1 = get_min_id(node_dict1)
        common_anc_id1 = list(get_node_dict_founder_set(node_dict1))[0]
    elif len(leaf_set1) == 1:
        min_id1 = list(leaf_set1)[0]
        common_anc_id1 = min_id1
    if node_dict2:
        min_id2 = get_min_id(node_dict2)
        common_anc_id2 = list(get_node_dict_founder_set(node_dict2))[0]
    elif len(leaf_set2) == 1:
        min_id2 = list(leaf_set2)[0]
        common_anc_id2 = min_id2

    presence_absence_dict1 = {uid : False for uid in leaf_set1}
    log_prob_tuple1 = get_ibd_pattern_log_prob(node=common_anc_id1, node_dict=node_dict1, ibd_presence_absence_dict=presence_absence_dict1)

    presence_absence_dict2 = {uid : False for uid in leaf_set2}
    log_prob_tuple2 = get_ibd_pattern_log_prob(node=common_anc_id2, node_dict=node_dict2, ibd_presence_absence_dict=presence_absence_dict2)

    T1 = 1 - np.exp(log_prob_tuple1[1])
    T2 = 1 - np.exp(log_prob_tuple2[1])

    k_hat = L_merged_tot / (GENOME_LENGTH * T1 * T2)

    theta = k_hat / 4

    king_bds = 1 / (2**np.array([d + 1/2 for d in range(0,14)])) # DRUID infers up to 13th degree

    est_deg = sum(king_bds > theta) - 1

    return max(0,est_deg)


def get_connecting_anc_pair_deg_Ltot_and_log_like(
    ca1 : int,
    ca2 : int,
    po1 : Any,
    po2 : Any,
    ibd_seg_list : List[List[Any]],
) -> Tuple[int,float,float]:
    """
    Get the most likely degree and log likelihood for connecting the pedigrees
    through the common ancestors specified by the ancestors of node_dict1 and
    node_dict2. The likelihood is based on the observed amounts of IBD in each of
    the leaves in leaf_set1 and leaf_set2.
    Args:
        L_tot : total amount of IBD shared between leaf_set1 and leaf_set2
        node_dict1: node dict corresponding to leaf_set1 : a dict of the form
                    { node : {desc1 : deg1, desc2 : deg2, ....} }
        node_dict2: node dict corresponding to leaf_set2 : a dict of the form
                    { node : {desc1 : deg1, desc2 : deg2, ....} }
        leaf_set1: set of leaves in one clade
        leaf_set2: set of leaves in the other clade
        unrel_leaf_set1: set of other leaves in pedidgree 1 who are unrelated to leaves in pedigree 2
        unrel_leaf_set2: set of other leaves in pedidgree 2 who are unrelated to leaves in pedigree 1
        ibd_seg_list: list of the form [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
        bgd_mean_ct : expected number of background IBD segments between unrelated pair
        bgd_mean_len : expected length of each background IBD segment between an unrelated pair
    """

    gt_desc_set1 = {uid for uid in po1.rel_dict[ca1]['desc'] if uid > 0}
    gt_desc_set2 = {uid for uid in po2.rel_dict[ca2]['desc'] if uid > 0}

    if ca1 > 0:
        gt_desc_set1.add(ca1)
    if ca2 > 0:
        gt_desc_set2.add(ca2)

    indt_gt_desc_set1 = po1.get_independent_inds(gt_desc_set1)
    indt_gt_desc_set2 = po2.get_independent_inds(gt_desc_set2)

    node_dict1 = get_node_dict(po1, indt_gt_desc_set1)
    node_dict2 = get_node_dict(po2, indt_gt_desc_set2)

    node_dict1 = adjust_node_dict_to_common_anc(ca1,po1,node_dict1)
    node_dict2 = adjust_node_dict_to_common_anc(ca2,po2,node_dict2)

    min_id1 = get_min_id(node_dict1)
    min_id2 = get_min_id(node_dict2)
    root_id = min(min_id1,min_id2) - 1

    chrom_ibd_segs_dict = get_ibd_segs_between_sets(indt_gt_desc_set1, indt_gt_desc_set2, ibd_seg_list)
    merged_chrom_ibd_segs_dict = merge_ibd_segs(chrom_ibd_segs_dict)
    merged_ibd_seg_lengths = get_segment_length_list(merged_chrom_ibd_segs_dict)
    L_merged_tot = sum(merged_ibd_seg_lengths)

    deg = infer_degree_generalized_druid(
        leaf_set1 = indt_gt_desc_set1,
        leaf_set2 = indt_gt_desc_set2,
        node_dict1 = node_dict1,
        node_dict2 = node_dict2,
        L_merged_tot = L_merged_tot,
    )

    if deg == 0 and ca1 > 0 and ca2 > 0:
        raise BuildFailException("DRUID inferring {} and {} are identical.".format(ca1,ca2))

    deg1 = int(np.floor(deg/2))
    deg2 = int(np.ceil(deg/2))
    node_dict = {root_id : {ca1 : deg1, ca2 : deg2}}
    node_dict.update(node_dict1)
    node_dict.update(node_dict2)
    log_prob_ibd = get_log_prob_ibd(node_dict, root_id, ca1, ca2, num_common_ancs=1) # always compute with 1 common anc because then deg = up + down
    prob_ibd = np.exp(log_prob_ibd)

    mean = prob_ibd * GENOME_LENGTH
    var,El,El2 =  get_var_total_length_approx(node_dict, indt_gt_desc_set1, indt_gt_desc_set2, root_id, ca1, ca2, num_common_ancs = 1)

    expected_count = mean / El

    if L_merged_tot > 0:
        log_like = get_log_like_total_length_normal(L_merged_tot, mean, var) + logsumexp([0,-expected_count],b=[1,-1])
    else:
        log_like = -expected_count

    return deg, L_merged_tot, log_like


def get_open_ancestor_set(
    po : Any,
    node_id : int,
    leaf_set : Set[int],
    other_leaf_set : Set[int],
    ibd_seg_list : List[List[Any]],
    threshold : float = 0.05,
    use_overlap : bool = True,
    require_descendant_match = True,
) -> Tuple[Set[int],Set[int],Set[int]]:
    """
    Find the set of ancestors of node_id with the same genotyped desendants and
        1. No parents (open_viable_anc_id_set)
        2. One parent (half_open_anc_id_set)
        3. One parent where the existing parent has no relative/ancestor whose
           IBD with the descendants of node_id overlaps with the set of IBD between
           the descendants of node_id and other_leaf_set
    Args:
        po: pedigree object in which node_id resides
        node_id: individual for whom to look for open ancestors
        leaf_set: set of independent genotyped nodes below node_id
        other_leaf_set: set of independent genotyped nodes in the other pedigree to 
                        which leaf_set are related.
        ibd_seg_list : list of the form [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
        threshold: fraction of the total IBD shared between leaf_set, other_leaf_set, and the
                   relatives/ancestors of an ancestor of node_id for us to say that it's
                   unlikely other_leaf_set is attached through that ancestor.
        use_overlap: Use overlapping IBD segments to determine if two branches might are
                     inconsistent with being placed one off of each of a pair of ancestral spouses.
        require_descendant_match: require that any open ancestor has the same deescendants as node_id
    """
    # first find all ancestral ids with the same genotyped descendants as node_id
    node_genotyped_desc_set = {uid for uid in po.rel_dict[node_id]['desc'] if uid > 0}
    anc_dict = po.get_ancestor_dict(node_id)
    viable_anc_id_set = set()
    for anc_id,deg in anc_dict.items():
        anc_genotyped_desc_set = {uid for uid in po.rel_dict[anc_id]['desc'] if uid > 0}
        anc_genotyped_desc_set -= set([node_id])
        if (anc_genotyped_desc_set == node_genotyped_desc_set) or (not require_descendant_match):
            viable_anc_id_set.add(anc_id)
    # next subset only to ancestors who have 0 or 1 parents
    open_viable_anc_id_set = set() # zero parents
    half_open_anc_id_set = set() # one parent
    for anc_id in viable_anc_id_set:
        parent_ids = [pid for pid in po.up_pedigree_dict.get(anc_id,[])[2:] if pid is not None]
        if len(parent_ids) == 0:
            open_viable_anc_id_set.add(anc_id)
        elif len(parent_ids) == 1:
            half_open_anc_id_set.add(anc_id)
    # next find subset half open viable anc ids to those where no IBD 
    # segments from their existing parental side overlap with segments from other_leaf_set
    half_open_viable_anc_id_set = set()
    del_id_set = set()
    for anc_id in half_open_anc_id_set:
        genotyped_anc_rel_set = {uid for uid in po.rel_dict[anc_id]['rel'] | po.rel_dict[anc_id]['anc'] if uid > 0}
        is_overlap = check_overlap(
            focal_id_set = leaf_set,
            rel_id_set1 = genotyped_anc_rel_set,
            rel_id_set2 = other_leaf_set,
            ibd_seg_list = ibd_seg_list,
            threshold = threshold,
        )
        if (not is_overlap) or (use_overlap == False):
            half_open_viable_anc_id_set.add(anc_id)
            del_id_set.add(anc_id)
    half_open_anc_id_set -= del_id_set # remove viable half open ids, leaving only ones with overlaps
    return (open_viable_anc_id_set, half_open_viable_anc_id_set, half_open_anc_id_set)


def get_open_ancestor_set_for_leaves(
    po : Any,
    leaf_set : Set[int],
    other_leaf_set : Set[int],
    ibd_seg_list : List[List[Any]],
    threshold : float=0.05,
    use_overlap : bool=True,
    require_descendant_match : bool = True,
) -> Tuple[Set[int],Set[int],Set[int]]:
    """
    Get open ancestor sets for a set of leaf nodes.
    Args:
        po: pedigree object in which node_id resides
        leaf_set: set of independent genotyped nodes for whom to find open ancestors
        other_leaf_set: set of independent genotyped nodes in the other pedigree to 
                        which leaf_set are related.
        ibd_seg_list : list of the form [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
        threshold: fraction of the total IBD shared between leaf_set, other_leaf_set,
                   and the relatives/ancestors of an ancestor of node_id for us to say
                   that it's unlikely other_leaf_set is attached through that ancestor.
        use_overlap: Use overlapping IBD segments to determine if two branches might are
                     inconsistent with being placed one off of each of a pair of ancestral spouses.
        require_descendant_match: require that any open ancestor has the same deescendants as node_id
    """
    common_anc_dict = po.get_common_ancestor_dict(list(leaf_set), get_mrcas=True)
    open_set : Set[int] = set()
    half_open_set : Set[int] = set()
    half_open_set_overlap : Set[int] = set()
    for anc_id in common_anc_dict.keys():
        anc_open_set, anc_half_open_set, anc_half_open_set_overlap = get_open_ancestor_set(
            po = po,
            node_id = anc_id,
            leaf_set = leaf_set,
            other_leaf_set = other_leaf_set,
            ibd_seg_list = ibd_seg_list,
            threshold = threshold,
            use_overlap = use_overlap,
            require_descendant_match = require_descendant_match,
        )
        open_set |= anc_open_set
        half_open_set |= anc_half_open_set
        half_open_set_overlap |= anc_half_open_set_overlap
    return (open_set, half_open_set, half_open_set_overlap)


def remove_symmetric_ancestors(
    open_set : Set[int],
    po : Any,
) -> Set[int]:
    """
    Find out if any pair of ids in open_set are spouses. Remove one of them if neither is
    genotyped. Because any pedigree created by joining through one ancestor will be the
    same as the pedigree created by joining through their spouse.
    Args:
        open_set: set of fully open ancestors obtained from get_open_ancestor_set() or
                  get_open_ancestor_set_for_leaves()
        po: pedigree object in which node_id resides
    """
    return_set = copy.copy(open_set) # don't modify arguments
    for uid,info in po.up_pedigree_dict.items():
        pid_set = {pid for pid in info[2:] if pid is not None}
        overlap_set = return_set & pid_set
        ungenotyped_overlap_set = {pid for pid in overlap_set if pid < 0}
        if len(ungenotyped_overlap_set) > 1:
            save_id = ungenotyped_overlap_set.pop()
            return_set -= ungenotyped_overlap_set
    return return_set


def get_connecting_founders_degs_and_log_likes(
    po1 : Any,
    po2 : Any,
    gt_set1 : Set[int],
    gt_set2 : Set[int],
    ibd_seg_list : List[List[Any]],
    threshold : float = 0.5,
    use_overlap : bool = True,
    require_descendant_match : bool = True,
) -> List[List[Any]]:
    """
    Cycle over all open ancestors. Make node dicts connecting po1 and po2 through 
    all possible common ancestors with all possible degrees.
    Return the set of most likey ancestor pairs and their degrees and log likelihoods, 
    where likelihoods are the gamma likelihoods computed
    between clades and do not include the full composite likelihoods.
        [(ca1,ca2,deg,log_like),(ca1,ca2,deg,log_like),...]
    Args:
        po1: pedgiree object 1
        po2: pedgiree object 2
        gt_set1: set of genotyped individuals in po1 for whom to find common ancestors
                 through with to connect them to po2
        gt_set2: set of genotyped individuals in po2 for whom to find common ancestors
                 through with to connect them to po1
        ibd_seg_list: list of the form [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
        threshold: in get_open_ancestor_set(), the fraction of the total IBD shared between
                   leaf_set, other_leaf_set, and the relatives/ancestors of an ancestor of
                   node_id for us to say that it's unlikely other_leaf_set is attached
                   through that ancestor.
        use_overlap: in get_open_ancestor_set(), use overlapping IBD segments to 
                     determine if two branches might are inconsistent 
                     with being placed one off of each of a pair of ancestral spouses.
        require_descendant_match: in get_open_ancestor_set(), require that any open 
                                  ancestor has the same deescendants as node_id
    """

    if (not gt_set1) or (not gt_set2):
        raise Exception("Leaf sets must be non-empty.")

    open_set1, half_open_set1, half_open_set_overlap1 = get_open_ancestor_set_for_leaves(
        po = po1,
        leaf_set = gt_set1,
        other_leaf_set = gt_set2,
        ibd_seg_list = ibd_seg_list,
        threshold = threshold,
        use_overlap = use_overlap,
        require_descendant_match = require_descendant_match,
    )
    open_set2, half_open_set2, half_open_set_overlap2 = get_open_ancestor_set_for_leaves(
        po = po2,
        leaf_set = gt_set2,
        other_leaf_set = gt_set1,
        ibd_seg_list = ibd_seg_list,
        threshold = threshold,
        use_overlap = use_overlap,
        require_descendant_match = require_descendant_match,
    )

    # remove symmetric ancestors from open sets
    open_set1 = remove_symmetric_ancestors(open_set1, po1)
    open_set2 = remove_symmetric_ancestors(open_set2, po2)

    # only use half open sets that produce overlaps if we need to.
    possible_anc_set1 = open_set1 | half_open_set1
    if not possible_anc_set1:
        possible_anc_set1 = half_open_set_overlap1

    possible_anc_set2 = open_set2 | half_open_set2
    if not possible_anc_set2:
        possible_anc_set2 = half_open_set_overlap2

    anc_deg_log_like_list : List[List[Any]] = list()
    for ca1 in possible_anc_set1:
        for ca2 in possible_anc_set2:
            deg,L_tot,log_like = get_connecting_anc_pair_deg_Ltot_and_log_like(
                ca1 = ca1,
                ca2 = ca2,
                po1 = po1,
                po2 = po2,
                ibd_seg_list = ibd_seg_list,
            )
            if L_tot > 0:
                anc_deg_log_like_list += [[ca1,ca2,deg,log_like]]

    return sorted(anc_deg_log_like_list, key = lambda x: -x[-1])


def get_deg1_deg2(
    deg : int,
    anc_id1 : int,
    anc_id2 : int,
    po1 : Any,
    po2 : Any,
    num_common_ancs : int = 2,
) -> Tuple[int,int,float]:
    """
    Use ages to determine deg1, the degree from the common ancestor of node_dict1 to the root ancestor,
    and deg2, the degree from the common ancestor of node_dict2 to the root ancestor.
    Args:
        deg: total inferred deg between anc_id1 and anc_id1
        anc_id1: id of common ancestor in pedigree 1
        anc_id2: id of common ancestor in pedigree 2
        po1: pedigree object 1
        po2: pedigree object 2
        num_common_ancs: number of common ancs of anc_id1 and anc_id2
    """
    gt_desc_deg_dict1 = {node_id : po1.rels[anc_id1][node_id][1] for node_id in po1.rel_dict[anc_id1]['desc'] if node_id > 0}
    gt_desc_deg_dict2 = {node_id : po2.rels[anc_id2][node_id][1] for node_id in po2.rel_dict[anc_id2]['desc'] if node_id > 0}
    if anc_id1 > 0:
        gt_desc_deg_dict1[anc_id1] = 0
    if anc_id2 > 0:
        gt_desc_deg_dict2[anc_id2] = 0

    gt_desc_deg_dict1 = {uid : deg for uid,deg in gt_desc_deg_dict1.items() if uid in po1.up_pedigree_dict and po1.up_pedigree_dict[uid][1] is not None}
    gt_desc_deg_dict2 = {uid : deg for uid,deg in gt_desc_deg_dict2.items() if uid in po2.up_pedigree_dict and po2.up_pedigree_dict[uid][1] is not None}

    # for a fixed degree and specified number of common ancestors, get the total up+down
    # number of meioses that is compatible with the degree.
    if num_common_ancs == 1:
        up_down_total = deg
    elif num_common_ancs == 2:
        up_down_total = deg + 1

    # set defaults to return if nothing is better
    est_deg1 = int(np.ceil(up_down_total/2))
    est_deg2 = up_down_total - est_deg1

    distributions = load_distributions()

    max_log_like = -INF
    for deg1 in range(up_down_total+1):
        deg2 = up_down_total - deg1
        log_like = 0.0
        for id1,desc_deg1 in gt_desc_deg_dict1.items():
            for id2,desc_deg2 in gt_desc_deg_dict2.items():
                new_deg_tuple = (deg1+desc_deg1, deg2+desc_deg2, num_common_ancs)
                age1 = po1.up_pedigree_dict[id1][1]
                age2 = po2.up_pedigree_dict[id2][1]
                pair_log_like_list = log_likelihoods_by_age(
                    [age1], 
                    [age2], 
                    new_deg_tuple, 
                    distributions.age_diff_moments
                )
                pair_log_like = pair_log_like_list[0]
                log_like += pair_log_like
        if log_like > max_log_like:
            max_log_like = log_like
            est_deg1 = deg1
            est_deg2 = deg2

    return (est_deg1,est_deg2,max_log_like)


def find_open_partner_and_update_po(
    node_id : int, 
    po : Any,
) -> Optional[int]:
    """
    WARNING: This function may modify the argument po

    Find a partner of node_id who is the unique partner sharing all their descendants
    and who has no genotyped relatives or ancestors. If none exists, try to add them.
    Args:
        node_id: id of individual for whom to find an open partner
        po: pedigree object in which node_id is placed
    """

    child_id_list = po.down_pedigree_dict.get(node_id,[])[2:]
    parent_id_set : Set[int] = set()
    some_child_with_one_parent = False
    for child_id in child_id_list:
        pids = po.up_pedigree_dict.get(child_id,[])[2:]
        parent_id_set |= set(pids) 
        if len(pids) < 2:
            some_child_with_one_parent = True
    partner_id_set = parent_id_set - {node_id}
    partner_id_set -= {None}

    # check if node_id has multiple partners.
    if len(partner_id_set) > 1:
        return None
    if len(partner_id_set) == 1 and some_child_with_one_parent:
        return None
    if len(partner_id_set) == 0 and len(child_id_list) > 1:
        return None

    if len(partner_id_set) == 1:
        partner_id = partner_id_set.pop()
        partner_gt_rel_set = {iid for iid in po.rel_dict[partner_id]['rel'] | po.rel_dict[partner_id]['anc'] if iid > 0}
        if partner_gt_rel_set: # if the partner has genotyped relatives, we shouldn't connect them to the other pedigree
            return None
        else:
            return partner_id
    else:
        pid = None # I would call this partner_id, but there seems to be a bug in flake8 that complains
        for cid in child_id_list:
            if not pid:
                pid = po.add_parent_for_child(child_id=cid)
            else:
                po.connect_parent_child(child_id=cid, parent_id=pid)
        return pid


def get_best_desc_id_set(
    gt_id_set : Set[int],
    other_gt_id_set : Set[int],
    po : Any,
    ibd_seg_list : List[List[Any]],
) -> Set[int]:
    """
    The gt_id_set may not have a single common ancestor due to background IBD/inbreeding etc.
    if necessary, find the common ancestor(s) whose descendants share the most with the other
    pedigree and retain their descendants.

    Args:
        gt_id_set: set of genotyped IDs in po who are related to ids in the other pedigree
        other_gt_id_set : set of genotyped ids in the other pedigree related to gt_id_set
        po : pedigree of interest
        ibd_seg_list: list of the form [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
    """
    best_gt_set = copy.copy(gt_id_set) # don't modify arguments
    common_anc_dict = po.get_common_ancestor_dict([*gt_id_set],get_mrcas=True)
    if not common_anc_dict:
        covering_anc_id_set = po.get_covering_ancestor_set(gt_id_set)
        L_tot_max = 0
        for anc_id in covering_anc_id_set:
            desc_id_set = po.rel_dict[anc_id]['desc'] & gt_id_set
            chrom_ibd_segs_dict = get_ibd_segs_between_sets(desc_id_set, other_gt_id_set, ibd_seg_list)
            merged_chrom_ibd_segs_dict = merge_ibd_segs(chrom_ibd_segs_dict)
            merged_ibd_seg_lengths = get_segment_length_list(merged_chrom_ibd_segs_dict)
            L_tot = sum(merged_ibd_seg_lengths)
            if L_tot > L_tot_max:
                L_tot_max = L_tot
                best_gt_set = desc_id_set
    return best_gt_set


def combine_pedigrees(
    po1 : Any,
    po2 : Any,
    ibd_seg_list : List[List[Any]],
    require_descendant_match : bool = True,
    threshold : float = 0.05,
    use_overlap : bool = True,
    num_peds : int = 4, 
    drop_ibd_alpha : float = 1e-4,
    ibd_stat_dict : Dict[FrozenSet[int],Dict[str,Any]] = None,
    pw_rels : Dict[int, Dict[int, Tuple[Any,Any,Any]]] = None,
    pw_log_likes : Dict[int, Dict[int, Dict[Tuple[Any,Any,Any], float]]] = None,
    disallow_distant_half_rels : bool = False,
) -> List[Any]:
    """
    Combine pedigrees po1 and po2 through the top most likely degrees and common ancestor pairs
    Args:
        po1: pedgiree object 1
        po2: pedgiree object 2
        ibd_seg_list: list of the form [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
        require_descendant_match: in get_open_ancestor_set(), require that any open 
                                  ancestor has the same deescendants as node_id
        threshold: in get_open_ancestor_set(), the fraction of the total IBD 
                   shared between leaf_set, other_leaf_set, and the relatives/ancestors
                   of an ancestor of node_id for us to say that it's unlikely other_leaf_set
                   is attached through that ancestor.
        use_overlap: in get_open_ancestor_set(), use overlapping IBD segments to determine if
                     two branches might are inconsistent  with being placed one off of each
                     of a pair of ancestral spouses.
        num_peds: return the top num_peds most likely pedigrees
        ibd_stat_dict : Dict mapping ID1 to ID2 to IBD summary stats between ID1 and ID2.
        pw_rels : Dict mapping ID1 to ID1 to a tuples representing their relationship (up,down,num_ancs)
        pw_log_likes : Dict mapping ID1 to ID1 to a dict mapping relationship tuples to their respective
                       point-predicted likelihoods.
    """

    min_id = min(po1.min_parent_ind, po2.min_parent_ind)
    id_update_dict = po2.update_ungenotyped_inds(min_id-1)

    gt_set1 = {node_id for node_id in po1.up_pedigree_dict.keys() if node_id > 0}
    gt_set2 = {node_id for node_id in po2.up_pedigree_dict.keys() if node_id > 0}
    
    gt_set1,gt_set2 = get_related_sets(gt_set1, gt_set2, ibd_seg_list)

    # gt_set1 may not have a single common ancestor due to background IBD/inbreeding etc.
    # if necessary, find the common ancestor(s) whose descendants share the most with pedigree 2
    # and retain their descendants as gt_set1
    gt_set1 = get_best_desc_id_set(gt_set1, gt_set2, po1, ibd_seg_list)
    gt_set2 = get_best_desc_id_set(gt_set2, gt_set1, po2, ibd_seg_list)

    anc_deg_log_like_list = get_connecting_founders_degs_and_log_likes(
        po1 = po1,
        po2 = po2,
        gt_set1 = gt_set1,
        gt_set2 = gt_set2,
        ibd_seg_list = ibd_seg_list,
        require_descendant_match = True,
        use_overlap = use_overlap,
    )

    num_arrangements = len(anc_deg_log_like_list)

    ind = 0
    new_ped_obj_list : List[Any] = list()
    while ind < num_arrangements and len(new_ped_obj_list) < num_peds:
        ca1,ca2,deg,log_like = anc_deg_log_like_list[ind]
        ind += 1

        ca1 = drop_background_ibd(
            ca1 = ca1,
            ca2 = ca2,
            po1 = po1,
            po2 = po2,
            ibd_seg_list = ibd_seg_list,
            drop_threshold = drop_ibd_alpha,
        )

        if not ca1:
            continue

        ca2 = drop_background_ibd(
            ca1 = ca2,
            ca2 = ca1,
            po1 = po2,
            po2 = po1,
            ibd_seg_list = ibd_seg_list,
            drop_threshold = drop_ibd_alpha,
        )

        if not ca2:
            continue

        gt_set1 = {iid for iid in po1.rel_dict[ca1]['desc'] if iid > 0}
        gt_set2 = {iid for iid in po2.rel_dict[ca2]['desc'] if iid > 0}

        gt_set1 |= {ca1} if ca1 > 0 else set()
        gt_set2 |= {ca2} if ca2 > 0 else set()

        new_anc_deg_log_like_list = get_connecting_founders_degs_and_log_likes(
            po1 = po1,
            po2 = po2,
            gt_set1 = gt_set1,
            gt_set2 = gt_set2,
            ibd_seg_list = ibd_seg_list,
            require_descendant_match = True,
            use_overlap = use_overlap,
        )

        if not new_anc_deg_log_like_list:
            continue
        else:
            ca1,ca2,deg,log_like = anc_deg_log_like_list[0]

        # Get degrees and number of common ancestors
        num_common_ancs = 2
        deg1,deg2,log_like = get_deg1_deg2(deg, ca1, ca2, po1, po2, num_common_ancs)
        if not disallow_distant_half_rels:
            deg1_1,deg2_1,log_like_1 = get_deg1_deg2(deg, ca1, ca2, po1, po2, num_common_ancs=1)
            if log_like_1 > log_like:
                deg1 = deg1_1
                deg2 = deg2_1
                num_common_ancs = 1

        po1_copy = copy.deepcopy(po1)
        po2_copy = copy.deepcopy(po2)

        if deg1 > 0 and len(po1_copy.up_pedigree_dict[ca1][2:]) > 0: # half open ancestor
            ca1 = po1_copy.add_parent_for_child(child_id=ca1)
            deg1 -= 1
        if deg2 > 0 and len(po2_copy.up_pedigree_dict[ca2][2:]) > 0: # half open ancestor
            ca2 = po2_copy.add_parent_for_child(child_id=ca2)
            deg2 -= 1

        ca1_partner_id = None
        ca2_partner_id = None
        if num_common_ancs == 2:
            if deg1 == 0:
                ca1_partner_id = find_open_partner_and_update_po(ca1, po1_copy) # modifies po1_copy
            if deg2 == 0:
                ca2_partner_id = find_open_partner_and_update_po(ca2, po2_copy) # modifies po1_copy

        new_ped_obj = connect_pedigrees_through_founders(
            anc_id1 = ca1,
            anc_id2 = ca2,
            po1 = po1_copy,
            po2 = po2_copy,
            deg1 = deg1,
            deg2 = deg2,
            partner_id1 = ca1_partner_id,
            partner_id2 = ca2_partner_id,
            ibd_stat_dict = ibd_stat_dict,
            pw_log_likes = pw_log_likes,
            num_common_ancs = num_common_ancs,
        )

        if new_ped_obj:

            consistent_pw_likes = True
            if pw_rels and pw_log_likes:
                gt_id_list = [uid for uid in new_ped_obj.up_pedigree_dict.keys() if uid > 0]
                age_dict = {uid : info[1] for uid,info in new_ped_obj.up_pedigree_dict.items() if uid > 0}
                consistent_pw_likes,unlikely_pair_info_list = check_pred_deg_likes(
                    pw_rels=pw_rels,
                    pw_log_likes=pw_log_likes,
                    ped_pw_rels=new_ped_obj.rels,
                    placed_gt_id_list=gt_id_list,
                    age_dict=age_dict,
                    focal_id=None,
                    radius_deg=float('inf'),
                    deg_diff=5,
                    throw_exception=False
                )
            inconsistent_sexes,inconsistent_sex_node = new_ped_obj.inconsistent_sexes()

            if consistent_pw_likes and (not inconsistent_sexes):
                new_ped_obj_list.append(new_ped_obj)

    new_ped_obj_list = sorted(new_ped_obj_list, key=lambda x: -x.pedigree_log_likelihood)

    return new_ped_obj_list


def find_closest_pedigrees(
    index_to_gtid_set : Dict[int,Set[int]],
    ibd_seg_list : List[List[Any]],
) -> List[int]:
    """
    Find the two pedigrees whose genotyped members share the most half IBD.
    Args:
        index_to_gtid_set : Dict mapping the index of the pedigree object to the set of
                            genotyped IDs in the object
        ibd_seg_list : list of the form [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
    """
    L_tot_max = 0
    closest_ped_idxs = []
    for idx1,idx2 in combinations(index_to_gtid_set.keys(), r=2):
        id_set1 = index_to_gtid_set[idx1]
        id_set2 = index_to_gtid_set[idx2]

        chrom_ibd_segs_dict = get_ibd_segs_between_sets(id_set1, id_set2, ibd_seg_list)
        merged_chrom_ibd_segs_dict = merge_ibd_segs(chrom_ibd_segs_dict)
        merged_ibd_seg_lengths = get_segment_length_list(merged_chrom_ibd_segs_dict)
        L_tot = sum(merged_ibd_seg_lengths)

        if L_tot > L_tot_max:
            L_tot_max = L_tot
            closest_ped_idxs = [idx1,idx2]

    return closest_ped_idxs


def drop_background_ibd(
    ca1 : int,
    ca2 : int,
    po1 : Any,
    po2 : Any,
    ibd_seg_list : List[List[Any]],
    drop_threshold : float,
    seen_ca1_set : Set[int] = None,
) -> Any:
    """
    Test whether any node below ca1 is the ancestor of a clade with unlikely amounts of
    shared IBD with the other pedigree. If so, drop it out and re-set ca1 to be the node
    below ca1 that does not have unlikely amounts of IBD. Repeat until both nodes below
    ca1 have reasonably likely amounts of IBD. If there is no node below, as in the case
    where ca1 is the parent of two genotyped siblings at least one of which has
    background ibd, return None.
    Args:
        ca1 : root of node_dict1
        ca2 : root of node_dict2
        node_dict1: independent node dict corresponding to 
                    leaf_set1 : a dict of the form { node : {desc1 : deg1, desc2 : deg2, ....} }.
                    ASSUME INDT
        node_dict2: independent node dict corresponding to 
                    leaf_set2 : a dict of the form { node : {desc1 : deg1, desc2 : deg2, ....} }.
                    ASSUME INDT
        ibd_seg_list : list of the form [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
        seen_ca1_set : set of common ancestors we have already tried. Prevents infinite recursion
                       when ca1 has a partner who is a viable common ancestor, but 
    """

    if not seen_ca1_set:
        seen_ca1_set = set()

    # avoid infinite recursions
    if ca1 in seen_ca1_set:
        return None
    
    seen_ca1_set.add(ca1)

    node_dict1 = get_node_dict_for_root(
        root_id = ca1,
        ped_obj = po1,
    )
    node_dict2 = get_node_dict_for_root(
        root_id = ca2,
        ped_obj = po2,
    )

    all_node_set1 = {*node_dict1} | set.union(*[{*v} for v in node_dict1.values()])

    if not node_dict1:
        return ca1
    elif ca1 in node_dict1 and len(node_dict1[ca1]) == 1:
        return ca1

    leaf_set1 = {i for i in get_leaf_set(node_dict1) if i > 0}
    leaf_set2 = {i for i in get_leaf_set(node_dict2) if i > 0}

    if ca1 > 0 and not leaf_set1:
        leaf_set1.add(ca1)
    if ca2 > 0 and not leaf_set2:
        leaf_set2.add(ca2)

    min_id1 = get_min_id(node_dict1)
    min_id2 = get_min_id(node_dict2)
    root_id = min(-1, min(min_id1,min_id2) - 1)  # id lower than any value in either node_dict

    child_id_set = {*node_dict1[ca1]}

    # Loop over all clades directly descendended from ca1 and for each, infer the degree
    # separating it from ca2. Store the L_tot between the clade and leaf_set2 along with
    # the inferred degree between ca1 and ca2 (adj_deg)
    child_id_count_L_tot_list = []
    safe_child = None # child node we do not drop. Shares most IBD with po2.
    L_tot_max = 0
    min_adj_deg = INF
    for child_id in child_id_set:

        child_node_dict = extract_down_node_dict(child_id,node_dict1)
        if child_node_dict:
            child_leaf_set = get_leaf_set(child_node_dict)
        else:
            child_leaf_set = {child_id}

        if not child_node_dict:
            child_node_dict = {child_id : {child_id : 0}}
            child_leaf_set = {child_id}

        chrom_ibd_segs_dict = get_ibd_segs_between_sets(child_leaf_set, leaf_set2, ibd_seg_list)
        merged_chrom_ibd_segs_dict = merge_ibd_segs(chrom_ibd_segs_dict)
        merged_ibd_seg_lengths = get_segment_length_list(merged_chrom_ibd_segs_dict)
        count = len(merged_ibd_seg_lengths)
        L_tot = sum(merged_ibd_seg_lengths)

        est_deg = infer_degree_generalized_druid(
            leaf_set1 = child_leaf_set,
            leaf_set2 = leaf_set2,
            node_dict1 = child_node_dict,
            node_dict2 = node_dict2,
            L_merged_tot = L_tot,
        )

        adj_deg = est_deg - node_dict1[ca1][child_id]

        if adj_deg < min_adj_deg:
            min_adj_deg = adj_deg
            safe_child = child_id
        elif adj_deg == min_adj_deg and L_tot > L_tot_max:
            safe_child = child_id

        L_tot_max = max(L_tot, L_tot_max)

        child_id_count_L_tot_list.append((child_id, count, L_tot))

    # Assuming the degree is ca1_ca2_est_deg, find clades with very low or high IBD
    drop_child_set = set()
    for child_id,count,L_tot in child_id_count_L_tot_list:

        if child_id == safe_child:
            continue

        child_node_dict = extract_down_node_dict(child_id,node_dict1)
        if child_id in child_node_dict: # if it's an internal node
            child_leaf_set = get_leaf_set(child_node_dict)
        else:
            child_leaf_set = {child_id}

        # re-infer the degree between the common ancestors, leaving out child_id
        # and their descendants so that the estimate is based on all other individuals
        child_desc_set = po1.rel_dict[child_id]['desc'] | {child_id}
        leave_one_out_leaf_set1 = leaf_set1 - child_desc_set

        # get total length of IBD shared between leave-one-out set and po2
        chrom_ibd_segs_dict = get_ibd_segs_between_sets(leave_one_out_leaf_set1, leaf_set2, ibd_seg_list)
        merged_chrom_ibd_segs_dict = merge_ibd_segs(chrom_ibd_segs_dict)
        merged_ibd_seg_lengths = get_segment_length_list(merged_chrom_ibd_segs_dict)
        leave_one_out_L_tot = sum(merged_ibd_seg_lengths)

        # get the node dict for the leave-one-out leaf set
        leave_one_out_common_anc_dict1 = po1.get_common_ancestor_dict([*leave_one_out_leaf_set1], get_mrcas=True)
        leave_one_out_common_anc1 = list({*leave_one_out_common_anc_dict1} & all_node_set1)[0]
        leave_one_out_node_dict1 = get_node_dict_for_root(
            root_id = leave_one_out_common_anc1,
            ped_obj = po1,
        )

        # infer the degree between the common ancestor of the leave-one-out set and ca2
        leave_one_out_ca1_ca2_est_deg = infer_degree_generalized_druid(
            leaf_set1 = leave_one_out_leaf_set1,
            leaf_set2 = leaf_set2,
            node_dict1 = leave_one_out_node_dict1,
            node_dict2 = node_dict2,
            L_merged_tot = leave_one_out_L_tot,
        )

        # adjust to get the degree between ca1 and ca2
        if leave_one_out_common_anc1 == ca1:
            ca1_ca2_est_deg = leave_one_out_ca1_ca2_est_deg
        else:
            ca1_ca2_est_deg = leave_one_out_ca1_ca2_est_deg - po1.rels[ca1][leave_one_out_common_anc1][1]

        # get the degree between child_id and ca2, and make a combined node dict 
        # for the background IBD hypothesis test
        deg = ca1_ca2_est_deg + node_dict1[ca1][child_id]
        deg1 = int(np.floor(deg/2))
        deg2 = int(np.ceil(deg/2))
        node_dict = {root_id : {child_id : deg1, ca2 : deg2}}
        node_dict.update(child_node_dict)
        node_dict.update(node_dict2)

        # Get mean and variance of total IBD shared between descendants of child ID and po2
        # always compute with 1 common anc because then deg = up + down
        log_prob_ibd = get_log_prob_ibd(node_dict, root_id, child_id, ca2, num_common_ancs=1)
        prob_ibd = np.exp(log_prob_ibd)
        mean = prob_ibd * GENOME_LENGTH
        var,El,El2 =  get_var_total_length_approx(node_dict, child_leaf_set, leaf_set2, root_id, child_id, ca2, num_common_ancs = 1)
        
        if mean > 0 and El > 0:
            expected_count = mean / El
        else:
            expected_count = 0

        # Get the background IBD pvalue
        pval = get_background_test_pval_gamma(L_tot,mean,var)

        if pval < drop_threshold:
            drop_child_set.add(child_id)

    new_ca1 = copy.copy(ca1)
    if drop_child_set and (len(child_id_set - drop_child_set) == 1):
        # if we rejected the null and exactly one child was not rejected, set them as the new ca1
        new_ca1 = (child_id_set - drop_child_set).pop()
    elif drop_child_set:
        # otherwise, if there are several children for whom we did not reject the null,
        # test whether the remaining children have a common ancestor other than ca1 who can be the new ca1
        # if not, return None

        # get all direct biological children of ca1 who eithe are, or are ancestral to an un-dropped descendant
        biological_child_id_set = {*po1.down_pedigree_dict[ca1][2:]} # biological children of ca1
        ancestral_biological_child_id_set = set() # ancestral to true ibd-sharing descendants
        for child_id in child_id_set - drop_child_set:
            ancestral_biological_child_id_set |= po1.rel_dict[child_id]['anc'] & biological_child_id_set
            ancestral_biological_child_id_set |= {child_id} & biological_child_id_set
        partner_id_set = set()

        # get all partners ca1 who are parents of children in ancestral_biological_child_id_set
        for biological_child_id in ancestral_biological_child_id_set:
            partner_id_set |= {i for i in po1.up_pedigree_dict[biological_child_id][2:] if i is not None}
        partner_id_set -= {ca1}

        if len(partner_id_set) == 1: # if more than one partner, then the common ancestor can only be ca1
            # but have to first check that no dropped child is descended from that partner
            dropped_child_from_partner = False
            for child_id in drop_child_set:
                if partner_id_set & po1.rel_dict[child_id]['anc']:
                    dropped_child_from_partner = True
            if not dropped_child_from_partner: # partner is valid common ancestor.
                new_ca1 = partner_id_set.pop()
            else: # some background IBD child is descended from the partner. So not a valid common ancestor
                return None
        else: # true IBD-carrying descendants of ca1 only have ca1 as a common ancestor
            return None # we could continue to check background IBD in these descendants, but it's hard. for now, return None.

    if new_ca1 == ca1: # if we did not drop any descendants
        return new_ca1
    else: # if we found a new potential common ancestor, check for background IBD below them
        return drop_background_ibd(
            ca1 = new_ca1,
            ca2 = ca2,
            po1 = po1,
            po2 = po2,
            ibd_seg_list = ibd_seg_list,
            drop_threshold = drop_threshold,
            seen_ca1_set = seen_ca1_set,
        )


def check_pred_deg_likes(
    focal_id,
    pw_rels,
    pw_log_likes,
    ped_pw_rels,
    placed_gt_id_list,
    age_dict,
    radius_deg=5,
    deg_diff=2,
    user_error_detected=False,
    throw_exception=False
):
    """
    Check whether all pairs closer than 1st cousins within a radius of first cousin 
    around focal_id have predicted bonsai degrees that are close to their ML degrees.
    Args:
        focal_id: id of the focal individual for whom we are building a pedigree
        pw_rels : Dict mapping ID1 to ID1 to a tuples representing their point-predicted
                  relationship (up,down,num_ancs)
        pw_log_likes : Dict mapping ID1 to ID1 to a dict mapping relationship tuples to
                       their respective point-predicted likelihoods. 
        ped_pw_rels : Dict mapping ID1 to ID1 to a tuples representing their bonsai-inferred
                      relationship (up,down,num_ancs)
        placed_gt_id_list : set of genotyped ids among which to check likelihoods
        age_dict: dict mapping id to age
        radius_deg: only consider ids whose degree to the focal_id is at most this
        deg_diff: say that the point-predicted and bonsai-inferred relationships between
                  two closely-related idnividuals are discordant if they differ by at
                  least this many degrees.
        user_error_detected: boolean specifying that the point predicte likelihood may be
                             incorrect (due to user error)
        throw_exception: if true, error hard when an unlikely relationship is detected.
    """
    unrel_deg = (float('inf'),float('inf'),None)
    parent_deg_set = {(1,0,1),(0,1,1)}

    bonsai_likelihood_agreement = True
    warn_only = True
    unlikely_pair_info_list = []
    for id1,id2 in combinations(placed_gt_id_list,r=2):

        # check pw degree of id1 and id2 to focal. If too far away, ignore 
        # so we don't fail on trees if a distant relationship is wrong.
        if focal_id is not None:
            if id1 == focal_id:
                deg1 = (0,0,2)
            else:
                deg1 = pw_rels[focal_id][id1]

            if id2 == focal_id:
                deg2 = (0,0,2)
            else:
                deg2 = pw_rels[focal_id][id2]

            if deg1 == unrel_deg:
                abs_deg1 = float('inf')
            else:
                abs_deg1 = deg1[0] + deg1[1] - deg1[2] + 1

            if deg2 == unrel_deg:
                abs_deg2 = float('inf')
            else:
                abs_deg2 = deg2[0] + deg2[1] - deg2[2] + 1

            if (abs_deg1 > radius_deg) or (abs_deg2 > radius_deg):
                continue

        # check degree between id1 and id2. Ignore if id1 and id2 are too far apart
        # so we don't fail on trees if a distant rel is wrong.
        pw_deg = pw_rels[id1][id2]
        pred_deg = ped_pw_rels[id1].get(id2,unrel_deg)
        
        if pw_deg == unrel_deg:
            abs_pw_deg = float('inf')
        else:
            abs_pw_deg = pw_deg[0] + pw_deg[1] - pw_deg[2] + 1

        if pred_deg == unrel_deg:
            abs_pred_deg = float('inf')
        else:
            abs_pred_deg = pred_deg[0] + pred_deg[1] - pred_deg[2] + 1

        if (abs_pw_deg > radius_deg) and (abs_pw_deg != float('inf')): # allow inf so we can test if unrelated are being placed together
            continue

        # check whether the original pw estimate may have been wrong
        ambiguous_pw_deg = False
        if pw_deg in parent_deg_set:
            age1 = age_dict.get(id1)
            age2 = age_dict.get(id2)
            if (not isinstance(age1,int)) or (not isinstance(age2,int)):
                ambiguous_pw_deg = True
            elif abs(age1 - age2) <= MIN_PARENT_CHILD_AGE_DIFF:
                ambiguous_pw_deg = True
        # can add more cases ...

        if ambiguous_pw_deg and {pw_deg,pred_deg} == parent_deg_set:
            continue
        elif ({pw_deg,pred_deg} & parent_deg_set) and pw_deg != pred_deg:
            bonsai_likelihood_agreement = False
            warn_only = False
            unlikely_pair_info_list.append([focal_id,id1,id2,pw_deg,pred_deg])
        elif (1,1,2) in {pw_deg,pred_deg} and pw_deg != pred_deg:
            bonsai_likelihood_agreement = False
            warn_only = False
            unlikely_pair_info_list.append([focal_id,id1,id2,pw_deg,pred_deg])
        elif pw_deg[2] is None:
            if abs_pred_deg <= 5:
                bonsai_likelihood_agreement = False
                unlikely_pair_info_list.append([focal_id,id1,id2,pw_deg,pred_deg])
                if abs_pred_deg <= 3:
                    warn_only = False # if it's super unlikely, actually raise an exception rather than a warning.
        elif abs(abs_pw_deg - abs_pred_deg) > deg_diff:
            if abs_pred_deg == float('inf'):
                if abs_pw_deg <= 5:
                    bonsai_likelihood_agreement = False
                    unlikely_pair_info_list.append([focal_id,id1,id2,pw_deg,pred_deg])
            else:
                bonsai_likelihood_agreement = False
                warn_only = False
                unlikely_pair_info_list.append([focal_id,id1,id2,pw_deg,pred_deg])

    if bonsai_likelihood_agreement:
        return (True,unlikely_pair_info_list)
    else:
        err_message = ''
        for unlikely_pair_info in unlikely_pair_info_list:
            err_message += '\n' + "Unlikely pair. focal_id={}, id1={}, id1={}, pw_deg={}, pred_deg={}".format(*unlikely_pair_info)
        if throw_exception:
            print(err_message)
            if warn_only or user_error_detected:
                warnings.warn(err_message,UnlikelyRelationshipWarning)
            else:
                raise UnlikelyRelationshipException(err_message)
        else:
            return (False,unlikely_pair_info_list)
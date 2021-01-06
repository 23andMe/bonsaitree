from typing import Any, Dict, List, Set, Tuple, FrozenSet, Optional

import copy
import numpy as np

from itertools import combinations
from collections import defaultdict

from .distributions import AUTO_GENOME_LENGTH, MIN_PARENT_CHILD_AGE_DIFF, SMALL_LOG_PROB

INF = float('inf')
UNREL_DEG = (INF,INF,None)
TWIN_MIN_GENOME_LENGTH = 0.95 * AUTO_GENOME_LENGTH

FULL_SIB_SUFFICIENT_IBD2 = 1000
FULL_SIB_IBD_BDS = {'half_ibd_lbd' : 1500, 'half_ibd_ubd' : 3300, 'full_ibd_lbd' : 300}

def is_twin_pair(
    total_half_len : float = None,
    total_full_len : float = None,
    sex1 : str = None,
    sex2 : str = None,
) -> bool:
    """
    Checks if two individuals are twins.
    Args:
        total_half_len : the total length of IBD1 shared between the two people
        total_full_len : the total length of IBD2 shared between the two people
        sex1 : sex of person 1 ('M' or 'F')
        sex2 : sex of person 2 ('M' or 'F')
    """

    if total_half_len is None:
        return False
    if total_full_len is None:
        return False

    if total_half_len < TWIN_MIN_GENOME_LENGTH:
        return False
    elif total_full_len < TWIN_MIN_GENOME_LENGTH:
        return False
    elif sex1 != sex2:
        return False
    else:
        return True


def find_twins(
    pw_rels : Dict[int, Dict[int,Tuple[Any,Any,Any]]],
    ibd_stat_dict : Dict[FrozenSet[int],Dict[str,Any]],
    sex_dict : Dict[int,str],
    age_dict : Dict[int,int],
) -> Tuple[Dict[int, Set[int]], Dict[int, int]]:
    """
    Find all twins and x-tuplets.

    Args:
        pw_rels : Dict mapping ID1 to ID1 to a tuples representing their relationship (up,down,num_ancs)
        ibd_stat_dict : Dict mapping ID1 to ID2 to IBD summary stats between ID1 and ID2.
        sex_dict : Dict mapping id to sex (sex is a string 'M' or 'F' or None)
        age_dict : Dict mapping id to age in years.
    Outputs: 
        twin_id_to_set_dict: Dict[int, int] : twin_id_to_set[uid] = twin_set_index
        twin_set_dict : Dict[int, Set[int]] : twin_sets[twin_set_index] = {id1,id2,...}
    """

    twin_set_dict : Dict[int, Set[int]] = {}
    twin_id_to_set_dict : Dict[int, int] = {}

    set_idx_ct = 0
    for id1,id2 in combinations(pw_rels.keys(), r=2):
        total_half_len = ibd_stat_dict[frozenset({id1,id2})]['total_half']
        total_full_len = ibd_stat_dict[frozenset({id1,id2})]['total_full']
        half_seg_count = ibd_stat_dict[frozenset({id1,id2})]['num_half']
        sex1 = sex_dict[id1]
        sex2 = sex_dict[id2]
        age1 = age_dict[id1]
        age2 = age_dict[id2]
        if is_twin_pair(total_half_len, total_full_len, sex1, sex2):
            set_idx1 = twin_id_to_set_dict.get(id1)
            set_idx2 = twin_id_to_set_dict.get(id2)
            if (set_idx1 is not None) and (set_idx2 is not None):
                if set_idx1 == set_idx2:
                    continue
                for iid in twin_set_dict[set_idx2]:
                    twin_id_to_set_dict[iid] = set_idx1
                    twin_set_dict[set_idx1].add(iid)
                del twin_set_dict[set_idx2]
            elif set_idx1 is not None:
                twin_id_to_set_dict[id2] = set_idx1
                twin_set_dict[set_idx1].add(id2)
            elif set_idx2 is not None:
                twin_id_to_set_dict[id1] = set_idx2
                twin_set_dict[set_idx2].add(id1)
            else: # Neither twin observed yet
                twin_id_to_set_dict[id1] = set_idx_ct
                twin_id_to_set_dict[id2] = set_idx_ct
                twin_set_dict[set_idx_ct] = {id1,id2}
                set_idx_ct += 1

    return (twin_set_dict, twin_id_to_set_dict)


def remove_twins(
    twin_set_dict : Dict[int, Set[int]],
    pw_rels : Dict[int, Dict[int, Tuple[Any,Any,Any]]],
    pw_log_likes : Dict[int, Dict[int, Dict[Tuple[Any,Any,Any], float]]],
    keep_set : Set[int] = None,
) -> Tuple[
        Dict[int, Set[int]], 
        Dict[int, int], 
        Dict[int, Dict[int, Tuple[Any,Any,Any]]], 
        Dict[int, Dict[int, Dict[Tuple[Any,Any,Any], float]]]
    ]:
    """
    Delete all but one person in a set of x-tuplets from pw_rels and pw_log_likes.
    Return a dictionary mapping the retained person to a set of their unretained x-tuplet. 
    Return a dictionary mapping each unretained individual to their retained twin.
    Return an new version of pw_rels excluding the unretained people.
    Return an new version of pw_log_likes excluding the unretained people.

    Args:
        twin_set_dict : dict mapping an index to each set of twins
        pw_rels : Dict mapping ID1 to ID1 to a tuples representing their relationship (up,down,num_ancs)
        pw_log_likes : Dict mapping ID1 to ID1 to a dict mapping relationship tuples to their
                       respective point-predicted likelihoods.
        keep_set : specify any individuals you want to keep
    """

    if not keep_set:
        keep_set = set()

    placed_twin_to_del_twin_set_dict = dict()
    del_twin_to_placed_twin_dict = dict()
    for set_idx, twin_set in twin_set_dict.items():
        twins_in_keep_set = keep_set & twin_set
        if twins_in_keep_set:
            placed_id = list(twins_in_keep_set)[0]
            unplaced_ids = twin_set - twins_in_keep_set
        else:
            placed_id = list(twin_set)[0]
            unplaced_ids = twin_set - {placed_id}
        if unplaced_ids:
            placed_twin_to_del_twin_set_dict[placed_id] = unplaced_ids
            for unplaced_id in unplaced_ids:
                del_twin_to_placed_twin_dict[unplaced_id] = placed_id

    new_pw_rels : Dict[int, Dict[int, Tuple[Any,Any,Any]]] = dict()
    new_pw_log_likes : Dict[int, Dict[int, Dict[Tuple[Any,Any,Any], float]]] = dict()
    for id1 in pw_rels.keys():
        if id1 in del_twin_to_placed_twin_dict:
            continue
        new_pw_rels[id1] = dict()
        new_pw_log_likes[id1] = dict()
        for id2 in pw_rels[id1].keys():
            if id2 in del_twin_to_placed_twin_dict:
                continue
            new_pw_rels[id1][id2] = pw_rels[id1][id2]
            new_pw_log_likes[id1][id2] = pw_log_likes[id1][id2]

    return (placed_twin_to_del_twin_set_dict, del_twin_to_placed_twin_dict, new_pw_rels, new_pw_log_likes)


def enforce_rel_types(
    pair_log_likes : Dict[Tuple[Any,Any,Any], float], # Modifies the input
    deg_set : Set[Tuple[Any,Any,Any]],
):
    """
    WARNING: modifies pair_log_likes. This is expected behavior.

    Set all entries of pw_log_likes to SMALL_LOG_PROB if they are not in deg_set
    Args:
        pair_log_likes : Dict mapping relationship tuples to their respective
                         point-predicted likelihoods for a given pair.
    """

    for deg,log_like in pair_log_likes.items():
        if deg not in deg_set:
            pair_log_likes[deg] = SMALL_LOG_PROB


def enforce_relationships(
    pw_rels : Dict[int, Dict[int, Tuple[Any,Any,Any]]],
    pw_log_likes : Dict[int, Dict[int, Dict[Tuple[Any,Any,Any], float]]],
    sex_dict : Dict[int,str],
    age_dict : Dict[int,int],
    ibd_stat_dict : Dict[FrozenSet[int],Dict[str,Any]],
) -> Tuple[
    Dict[int, Dict[int, Tuple[Any,Any,Any]]],
    Dict[int, Dict[int, Dict[Tuple[Any,Any,Any], float]]]
]:
    """
    Enforce sibling and parent-child relationships when they can be corroborated by another
    sibling (or parent). Also detect full sib relationships we might have mis-inferred as
    half sib relationships. This helps to speed up pedigree building.

    Args:
        pw_rels : Dict mapping ID1 to ID1 to a tuples representing their relationship (up,down,num_ancs)
        pw_log_likes : Dict mapping ID1 to ID1 to a dict mapping relationship tuples to their
                       respective point-predicted likelihoods.
        sex_dict : Dict mapping id to sex (sex is a string 'M' or 'F' or None)
        age_dict : Dict mapping id to age in years.
        ibd_stat_dict : Dict mapping ID1 to ID2 to IBD summary stats between ID1 and ID2.
    """

    new_pw_rels = copy.deepcopy(pw_rels)
    new_pw_log_likes = copy.deepcopy(pw_log_likes)

    # Find putative parent-child and sibling pairs
    parent_degs = set({(0,1,1),(1,0,1)})
    sibling_degs = set({(1,1,2)})
    half_sibling_deg = (1,1,1)
    putative_parents : Dict[int,Set[int]] = defaultdict(set)
    putative_children : Dict[int,Set[int]] = defaultdict(set)
    putative_siblings : Dict[int,Set[int]] = defaultdict(set)
    putative_half_siblings : Dict[int,Set[int]] = defaultdict(set)
    for id1,rel_info in pw_rels.items():
        id1_age = age_dict.get(id1)
        id1_sex = sex_dict.get(id1)
        if (not id1_age) or (not id1_sex):
            continue
        for id2,rel_deg in rel_info.items():
            id2_age = age_dict.get(id2)
            id2_sex = sex_dict.get(id2)
            if (not id2_age) or (not id2_sex):
                continue
            if rel_deg in parent_degs:
                if id1_age - id2_age >= MIN_PARENT_CHILD_AGE_DIFF:
                    putative_parents[id2].add(id1)
                    putative_children[id1].add(id2)
                elif id2_age - id1_age >= MIN_PARENT_CHILD_AGE_DIFF:
                    putative_parents[id1].add(id2)
                    putative_children[id2].add(id1)
            elif rel_deg in sibling_degs:
                putative_siblings[id1].add(id2)
                putative_siblings[id2].add(id1)
            elif rel_deg == half_sibling_deg:
                putative_half_siblings[id1].add(id2)
                putative_half_siblings[id2].add(id1)

    # Enforce parent-child relationship if putatively parent-child and there's at
    # least one sibling with a putative parent-child relationship to back it up
    for id1,parent_set in putative_parents.items():
        id1_sibling_set = putative_siblings[id1]
        for id1_parent in list(parent_set):
            id1_parent_children = putative_children[id1_parent]
            true_sib_set = id1_sibling_set & id1_parent_children
            if len(true_sib_set) > 0:
                enforce_rel_types(pair_log_likes=new_pw_log_likes[id1][id1_parent], deg_set=set({(1,0,1)}))
                enforce_rel_types(pair_log_likes=new_pw_log_likes[id1_parent][id1], deg_set=set({(0,1,1)}))
                new_pw_rels[id1][id1_parent] = (1,0,1)
                new_pw_rels[id1_parent][id1] = (0,1,1)

        # Find unrelated people who both have an inferred parent-child relationship to another person
        # Enforce those parent-child relationships
        for p1,p2 in combinations(list(parent_set), r=2):
            if new_pw_rels[p1][p2] != (float('inf'),float('inf'),None): # Make sure they're inferred to be unrelated
                continue
            p1_children = putative_children[p1]
            p2_children = putative_children[p2]
            common_children = p1_children & p2_children
            for common_child in list(common_children):
                enforce_rel_types(pair_log_likes=new_pw_log_likes[common_child][p1], deg_set=set({(1,0,1)}))
                enforce_rel_types(pair_log_likes=new_pw_log_likes[p1][common_child], deg_set=set({(0,1,1)}))
                enforce_rel_types(pair_log_likes=new_pw_log_likes[common_child][p2], deg_set=set({(1,0,1)}))
                enforce_rel_types(pair_log_likes=new_pw_log_likes[p2][common_child], deg_set=set({(0,1,1)}))
                new_pw_rels[common_child][p1] = (1,0,1)
                new_pw_rels[p1][common_child] = (0,1,1)
                new_pw_rels[common_child][p2] = (1,0,1)
                new_pw_rels[p2][common_child] = (0,1,1)

    # Enforce siblings
    for id1,sib_set in putative_siblings.items():
        for id2 in list(sib_set):
            half_ibd_len = ibd_stat_dict[frozenset({id1,id2})]['total_half']
            full_ibd_len = ibd_stat_dict[frozenset({id1,id2})]['total_full']
            if full_ibd_len > 1000 or (half_ibd_len > FULL_SIB_IBD_BDS['half_ibd_lbd'] and half_ibd_len < FULL_SIB_IBD_BDS['half_ibd_ubd'] and full_ibd_len > FULL_SIB_IBD_BDS['full_ibd_lbd']):
                enforce_rel_types(pair_log_likes=new_pw_log_likes[id1][id2], deg_set=set({(1,1,2)}))
                enforce_rel_types(pair_log_likes=new_pw_log_likes[id2][id1], deg_set=set({(1,1,2)}))
                new_pw_rels[id1][id2] = (1,1,2)
                new_pw_rels[id2][id1] = (1,1,2)

    # Find half sibs who are probably really full sibs
    for id1,half_sib_set in putative_half_siblings.items():
        for id2 in half_sib_set:
            half_ibd_len = ibd_stat_dict[frozenset({id1,id2})]['total_half']
            full_ibd_len = ibd_stat_dict[frozenset({id1,id2})]['total_full']
            if half_ibd_len > FULL_SIB_IBD_BDS['half_ibd_lbd'] and half_ibd_len < FULL_SIB_IBD_BDS['half_ibd_ubd'] and full_ibd_len > FULL_SIB_IBD_BDS['full_ibd_lbd']:
                enforce_rel_types(pair_log_likes=new_pw_log_likes[id1][id2], deg_set=set({(1,1,2)}))
                enforce_rel_types(pair_log_likes=new_pw_log_likes[id2][id1], deg_set=set({(1,1,2)}))
                new_pw_rels[id1][id2] = (1,1,2)
                new_pw_rels[id2][id1] = (1,1,2)

    # find parent child and full sib sets we may have missed using IBD alone and enforce them
    for id1,id2 in combinations({*sex_dict}, r=2):
        key = frozenset({id1,id2})
        ibd_stats = ibd_stat_dict[key]
        half_ibd_len = ibd_stats['total_half']
        full_ibd_len = ibd_stats['total_full']
        if half_ibd_len > TWIN_MIN_GENOME_LENGTH and full_ibd_len < FULL_SIB_IBD_BDS['full_ibd_lbd'] - 100:
            if age_dict[id1] - age_dict[id2] >= MIN_PARENT_CHILD_AGE_DIFF:
                pid = id1
                cid = id2
            elif age_dict[id2] - age_dict[id1] >= MIN_PARENT_CHILD_AGE_DIFF:
                pid = id2
                cid = id1
            else:
                continue
            enforce_rel_types(pair_log_likes=new_pw_log_likes[cid][pid], deg_set=set({(1,0,1)}))
            enforce_rel_types(pair_log_likes=new_pw_log_likes[pid][cid], deg_set=set({(0,1,1)}))
            new_pw_rels[cid][pid] = (1,0,1)
            new_pw_rels[pid][cid] = (0,1,1)
        elif half_ibd_len > FULL_SIB_IBD_BDS['half_ibd_lbd'] and half_ibd_len < FULL_SIB_IBD_BDS['half_ibd_ubd'] and full_ibd_len > FULL_SIB_IBD_BDS['full_ibd_lbd']:
            enforce_rel_types(pair_log_likes=new_pw_log_likes[id1][id2], deg_set=set({(1,1,2)}))
            enforce_rel_types(pair_log_likes=new_pw_log_likes[id2][id1], deg_set=set({(2,1,2)}))
            new_pw_rels[id1][id2] = (1,1,2)
            new_pw_rels[id2][id1] = (1,1,2)

    return new_pw_rels, new_pw_log_likes
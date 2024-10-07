from typing import Any, Dict, List, Set, Tuple

import copy
from itertools import combinations

from .exceptions import MissingNodeException

INF = float('inf')


def get_node_dict(
    ped_obj : Any,
    rels : Set[int],
) -> Dict[int, Dict[int, int]]:
    """
    Get a dict of the form {node : {desc1 : deg1, desc2 : deg2, ....}} for all
    nodes ancestral to those in the set rels. The node dict ignores nodes that
    have only one descendant.

    Args:
        ped_obj: pedigree object in which to build the node dict
        rels: build a node dict for all individuals in rels and their ancesetors

    Returns:
        node_dict: dict of the form {node : {desc1 : deg1, desc2 : deg2, ....}}
    """
    indep_rels = ped_obj.get_independent_inds(rels) #Get a subset of rels, none of whom is descended from any other individual and who are the most ancestral
    if len(indep_rels) == 1:
        node = list(indep_rels)[0]
        node_dict = {node : {node : 0}}
        max_leaf_set = {node}
    else:
        indep_rels = indep_rels
        mrca_set : Set[int] = set() # all potential internal nodes of the node dict (excludes leaf nodes, i.e., indep_rels)
        for id1,id2 in combinations(rels, r=2):
            common_anc_dict = ped_obj.get_common_ancestor_dict(resid_list=[id1,id2], get_mrcas=True)
            mrca_set |= {*common_anc_dict}
        internal_node_set = set()
        for node_id in mrca_set: # exclude common ancs (spouses) who are unrelated to the relative set
            rel_set = ped_obj.rel_dict[node_id]['rel']
            desc_set = ped_obj.rel_dict[node_id]['desc']
            if rel_set & indep_rels:
                internal_node_set.add(node_id)
            elif indep_rels == desc_set & indep_rels:
                internal_node_set.add(node_id)
        node_dict = dict()
        for node_id in internal_node_set:
            desc_set = ped_obj.rel_dict[node_id]['desc']
            leaf_desc_set = desc_set & indep_rels # ids in indep_rels that are descended from node_id
            internal_node_desc_set = desc_set & internal_node_set # ids in internal_node_set that are descended from node_id
            indep_internal_node_desc_set = ped_obj.get_independent_inds(internal_node_desc_set) # set of top-level internal nodes descended from node_id
            # remove leaves who are under the top-level internal node descendants
            for internal_desc_node_id in indep_internal_node_desc_set:
                internal_desc_node_desc_set = ped_obj.rel_dict[internal_desc_node_id]['desc']
                leaf_desc_set -= internal_desc_node_desc_set
            # map node_dict to its top-level internal node descendants
            deg_dict = dict()
            for internal_desc_node_id in indep_internal_node_desc_set:
                deg = ped_obj.rels[node_id][internal_desc_node_id][1]
                deg_dict[internal_desc_node_id] = deg
            # map node_dict to all leaf nodes that are not under one of its top-level internal node descendants (i.e., leaf nodes for which it's a direct most recent common ancestor)
            for leaf_desc_id in leaf_desc_set:
                deg = ped_obj.rels[node_id][leaf_desc_id][1]
                deg_dict[leaf_desc_id] = deg
            node_dict[node_id] = deg_dict

    return node_dict


def get_node_dict_for_root(
    root_id : int,
    ped_obj : Any,
) -> Dict[int, Dict[int, int]]:
    """
    Get the node dict corresponding to all descendants of root_id
    Args:
        root_id: id of root of node dict
        ped_obj: pedigree object in which root_id is placed
    """

    desc_set = {uid for uid in ped_obj.rel_dict[root_id]['desc'] if uid > 0}
    if not desc_set:
        return {root_id : {root_id : 0}}
    node_dict = get_node_dict(ped_obj, desc_set)
    return adjust_node_dict_to_common_anc(
        anc_id = root_id,
        po = ped_obj,
        node_dict = node_dict,
    )


def extract_down_node_dict(
    node : int,
    node_dict : Dict[int, Dict[int, int]],
    sub_dict : Dict[int, Dict[int, int]] = None,
) -> Dict[int, Dict[int, int]]:
    """
    WARNING: modifies sub_dict, but this is designed to be called with sub_dict = None.

    Extract the sub node_dict with entries below "node"

    Args:
        node: id of root of node dict
        node_dict: existing node_dict we are subsetting
    """

    if sub_dict is None:
        all_node_set = {*node_dict}
        for val in node_dict.values():
            all_node_set |= {*val}
        if node not in all_node_set:
            raise MissingNodeException("Node {} is not in node_dict.".format(node))
        sub_dict = dict()

    child_dict = node_dict.get(node,{})
    if child_dict:
        sub_dict[node] = child_dict
    for child in child_dict.keys():
        if child != node:
            extract_down_node_dict(child,node_dict,sub_dict)
    return sub_dict


def get_leaf_set(
    node_dict : Dict[int, Dict[int, int]],
) -> Set[int]:
    """
    Find all the leaves in the node dict

    Args:
        node_dict: node dict for whic we are getting the leaf set
    """
    leaf_set = set()
    for node,child_dict in node_dict.items():
        for child,deg in child_dict.items():
            if child not in node_dict:
                leaf_set.add(child)
            elif child == node: # orphan node
                leaf_set.add(child)
    return leaf_set


def get_root_to_desc_degrees(
    root_id : int,
    node_dict : Dict[int, Dict[int, int]],
    root_deg : int = 0,
) -> Dict[int,int]:
    """
    Get a dict with keys given by leaf ids and values given by the
    degree from the root to the descendant id.

    Args:
        root_id: id of root of node dict
        node_dict: existing node_dict in which root_id is placed (not
                   necessarily as the root of the full node dict)
    """
    out_deg_dict = dict()
    out_deg_dict.update({root_id : root_deg})
    if root_id in node_dict:
        for node_id,deg in node_dict[root_id].items():
            if node_id != root_id:
                desc_deg_dict = get_root_to_desc_degrees(node_id, node_dict, root_deg + deg)
                out_deg_dict.update(desc_deg_dict)
    return out_deg_dict


def get_desc_deg_dict(
    anc_id : int,
    node_dict : Dict[int, Dict[int, int]],
) -> Dict[int, int]:
    """
    Return a dict with keys given by the descendants of anc_id
    and values given by the degrees from anc_id down to them

    Args:
        anc_id: id of the ancestral individual
        node_dict: a dict of the form { node : {desc1 : deg1, desc2 : deg2, ....} }
                   node_dict skips omits nodes that are ancestral to only one person.
    """

    deg_dict = {anc_id : 0}

    if anc_id in node_dict.get(anc_id,dict()):
        return deg_dict

    for child_id,deg in node_dict.get(anc_id,{}).items():
        desc_deg_dict = get_desc_deg_dict(
            anc_id = child_id,
            node_dict = node_dict,
        )
        desc_deg_dict = {k : v + deg for k,v in desc_deg_dict.items() if k != anc_id}
        deg_dict.update(desc_deg_dict)

    return deg_dict


def get_node_dict_founder_set(
    node_dict : Dict[int, Dict[int, int]],
) -> Set[int]:
    """
    Find the set of founders in a node dict

    Args:
        node_dict: node dict for which we want to find the founder set.
    """

    possible_ancs = {*node_dict}
    if len(possible_ancs) > 1:
        for node,child_dict in node_dict.items():
            for child_id in child_dict:
                if child_id in possible_ancs:
                    possible_ancs.remove(child_id)
    return possible_ancs


def get_min_id(
    node_dict : Dict[int, Dict[int, int]],
) -> int:
    """
    Find the lowest value of a node in node_dict

    Args:
        node_dict: node dict for which we ant to find the id with the lowest value
    """

    min_id = min(node_dict.keys())
    for parent_id,info in node_dict.items():
        min_kid_id = min(info.keys())
        if min_kid_id < min_id:
            min_id = min_kid_id
    return min_id


def adjust_node_dict_to_common_anc(
    anc_id : int,
    po : Any,
    node_dict : Dict[int, Dict[int, int]],
) -> Dict[int, Dict[int, int]]:
    """
    Adjust node_dict so that there is only one common ancestor and it's anc_id
    
    Args:
        anc_id: new root of the node_dict
        node_dict: node dict we are subsetting or extending
        po: object of the pedigree in which node_dict represents
            the relationship among all or a subset of nodes.
    """

    if anc_id in node_dict:
        return extract_down_node_dict(anc_id, node_dict)
    else:
        node_dict_common_anc_set = get_node_dict_founder_set(node_dict)
        desc_anc_id = list(po.rel_dict[anc_id]['desc'] & node_dict_common_anc_set)[0]
        del_anc_id_set = node_dict_common_anc_set - {desc_anc_id}
        down_deg = po.rels[anc_id][desc_anc_id][1]

        return_dict = dict()
        return_dict[anc_id] = {desc_anc_id : down_deg}
        down_dict = extract_down_node_dict(desc_anc_id, node_dict)
        return_dict.update(down_dict)
        return return_dict

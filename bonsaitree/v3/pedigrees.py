import copy
from itertools import combinations
from typing import Any, Optional

import numpy as np

from .caching import (
    unfreeze_dict,
)
from .constants import (
    INF,
    MAX_CON_PTS,
)
from .exceptions import BonsaiError
from .relationships import (
    get_deg,
    reverse_rel,
)
from .utils import get_pairs


def reverse_node_dict(
    dct : dict[int, dict[int, int]],
):
    """
    Reverse a node dict. If it's a down dict make it an up dict
    and vice versa.

    up_node_dict : {i : {anc1 : deg1, ...}, ...}
    down_node_dict : {i : {desc1 : deg1, ...}, ...}
    """

    rev_dct = {}
    for i,info in dct.items():
        for a,d in info.items():
            if a not in rev_dct:
                rev_dct[a] = {}
            rev_dct[a][i] = d

    return rev_dct


def get_subdict(
    dct : dict[int, dict[int, int]],
    node : int,
):
    """
    If dct is an "up" node dict, get
    the cone above node.

    If dct is a "down" node dict, get
    the descendant cone below node.

    Args:
        dct: Node dict of the form
                   {id : {parent/child1 : deg1, parent/child2 : deg2, ...}, ...}
        node: int node for whom we want to extract the subdict

    Returns:
        sub_dct: Dict of the form
                  {id : {parent/child1 : deg1, parent/child2 : deg2, ...}, ...}
                  with node as the root and only containing
                  nodes above/below node.
    """
    if node not in dct:
        return {}

    sub_dct = {}
    sub_dct[node] = copy.deepcopy(dct[node])

    for n in dct[node]:
        n_dct = get_subdict(dct, n)
        if n_dct:
            sub_dct.update(n_dct)

    return sub_dct


def get_sib_set(
    up_dct : dict[int, dict[int, int]],
    down_dct : dict[int, dict[int, int]],
    node : int,
):
    """
    Get all sibling nodes of node

    Args:
        up_dct: Node dict of the form
                   {id : {parent1 : deg1, parent2 : deg2, ...}, ...}
        down_dct: Node dict of the form
                   {id : {child1 : deg1, child2 : deg2, ...}, ...}
        node: node to unfurl on

    Returns:
        sib_set: set of all sibs (half or full)
    """
    if node not in up_dct:
        return set()

    item = up_dct[node]

    pid_set = {*item}

    sib_set = set()
    for pid in pid_set:
        pid_child_set = {*down_dct[pid]}
        sib_set |= pid_child_set

    sib_set -= {node}

    return sib_set


def get_unfolded_up_node_dict(
    up_dct : dict[int, dict[int, int]],
    down_dct : dict[int, dict[int, int]],
    node : int,
):
    """
    Take in an up node dict and a node
    and return the node dict of all ancestral
    and collateral relatives obtained by "folding"
    up the node dict above node as if it were
    an umbrella blown inside out.

    Specifically, unf_dct is the node dict
    comprised of the focal cone of node, but
    with the descendant node dict of each ancestor
    added.

    Args:
        up_dct: Node dict of the form
                   {id : {parent1 : deg1, parent2 : deg2, ...}, ...}
        down_dct: Node dict of the form
                   {id : {child1 : deg1, child2 : deg2, ...}, ...}
        node: node to unfurl on

    Returns:
        unf_dct: unfolded node dict
    """

    unf_dct = {}
    if node in up_dct:
        unf_dct[node] = up_dct[node]

    for pid in up_dct.get(node, {}):
        if pid not in unf_dct:
            unf_dct[pid] = {}
        for cid,deg in down_dct[pid].items():
            if cid != node:
                unf_dct[pid][cid] = deg
                sub_dct = get_subdict(
                    dct=down_dct,
                    node=cid,
                )
                unf_dct.update(sub_dct)

        pid_unf_dct = get_unfolded_up_node_dict(
            up_dct=up_dct,
            down_dct=down_dct,
            node=pid
        )

        for k,v in pid_unf_dct.items():
            if k not in unf_dct:
                unf_dct[k] = v
            else:
                unf_dct[k].update(v)

    return unf_dct


# @freeze_args
# @functools.cache
def re_root_up_node_dict(
    up_dct : dict[int, dict[int, int]],
    node : int,
):
    """
    Re-root an up node dict on a particular node

    Args:
        up_dct: Node dict of the form
                   {id : {parent1 : deg1, parent2 : deg2, ...}, ...}
        node: node to reroot on.

    Returns:
        r_dct: down node dict rooted at node
    """
    up_dct = copy.deepcopy(unfreeze_dict(up_dct))
    # up_dct = copy.deepcopy(up_dct)

    down_dct = reverse_node_dict(up_dct)

    unf_dct = get_unfolded_up_node_dict(
        up_dct=up_dct,
        down_dct=down_dct,
        node=node,
    )

    sub_dct = get_subdict(
        dct=down_dct,
        node=node,
    )

    if sub_dct and unf_dct:
        # save time: find the smaller dict to copy into the larger
        if len(sub_dct) < len(unf_dct):
            r_dct = unf_dct
            s_dct = sub_dct
        else:
            r_dct = sub_dct
            s_dct = unf_dct

        # copy the the smaller dict into the larger
        r_dct[node].update(s_dct[node])
        for k,v in s_dct.items():
            if k != node:
                r_dct[k] = v
    elif sub_dct:
        r_dct = sub_dct
    else:
        r_dct = unf_dct

    return r_dct


def trim_to_proximal(
    down_dct : dict[int, dict[int, int]],
    root : int,
):
    """
    Get the down dict below root noode root and
    terminate the down branches at its most proximal genotyped
    nodes. Specifically if root R has descendant
    D, who in turn has desdcendants d1, d2, ...
    then trim_to_proximal() returns a down dict represnting
    the clade including and below R in
    which d1, d2, ... are removed along with the
    full clade below D (but not including D).


                    R
                ____|____
                |        |
               -1        -2
            ____|__      |
            |     |      |
            D     -3    -4
        ____|__
        |      |
        d1     -5

    Returns

                R
                |
               -1
                |
                D

    Args:
        down_dct: Node dict of the form
                   {id : {desc1 : deg1, desc2 : deg2, ...}, ...}
        root: root node of down_dct

    Returns:
        down_dct: modified so that it is truncated at
                  the most proximal genotyped nodes to root
    """

    if root not in down_dct:
        return {}

    trim_dct = {}

    trim_dct[root] = {}

    for child_id, deg in down_dct[root].items():
        if child_id > 0:
            trim_dct[root][child_id] = deg
        else:
            c_trim_dct = trim_to_proximal(down_dct, child_id)
            if c_trim_dct:
                trim_dct[root][child_id] = deg
                trim_dct.update(c_trim_dct)

    if trim_dct[root]:
        return trim_dct
    else:
        return {}


def get_proximal(
    node: int,
    up_dct: dict[int, dict[int, int]],
):
    """
    find the set of genotyped nodes that
    are proximal to node.

    Args:
        node: a node
        up_dct: Node dict of the form
                   {id : {parent1 : deg1, parent2 : deg2, ...}, ...}

    Returns:
        prox_set: set of genotyped nodes that
            have no other genotyped node on the
            path between them and node, including
            node itself.
    """
    if node > 0:
        return {node}

    dn_dct = re_root_up_node_dict(
        up_dct=up_dct,
        node=node,
    )
    dn_dct = trim_to_proximal(
        down_dct=dn_dct,
        root=node,
    )

    return get_gt_id_set(dn_dct)



def get_leaf_set_for_down_dct(
    down_dct : dict[int, dict[int, int]],
):
    """
    Get the leaf set from a down node dict

    Args:
        down_dct: Node dict of the form
                   {id : {desc1 : deg1, desc2 : deg2, ...}, ...}

    Returns:
        leaf_set: set of leaf nodes of the form {l1, l2,...}
    """

    up_dct = reverse_node_dict(down_dct)

    leaf_set = {*up_dct} - {*down_dct}

    return leaf_set


def get_leaf_set_for_up_dct(
    up_dct : dict[int, dict[int, int]],
):
    """
    Get the leaf set from a up node dict

    Args:
        up_dct: Node dict of the form
                   {id : {parent1 : deg1, parent2 : deg2, ...}, ...}

    Returns:
        leaf_set: set of leaf nodes of the form {l1, l2,...}
    """

    down_dct = reverse_node_dict(up_dct)

    leaf_set = {*up_dct} - {*down_dct}

    return leaf_set


def get_one_way_paths(
    node_dict : dict[int, dict[int, int]],
    i : int,
):
    """
    Find all paths extending upward/downward from i.

    if node_dict is an up_node_dict, find all paths extending upward

    if node_dict is a down_node_dict, find all paths extending downward
    """

    if i not in node_dict:
        return [[i]]
    if node_dict[i] == {}:
        return [[i]]

    path_list = []
    for j in node_dict[i]:
        j_paths = get_one_way_paths(node_dict, j)
        j_paths = [[i] + p for p in j_paths]
        path_list += j_paths

    return path_list


def get_rel_set(
    node_dict : dict[int, dict[int, int]],
    i : int,
):
    """
    If node_dict is an up_dict, get a set of all ancestors of i
    (including i)

    If node_dict is a down_dict, get the set of all descendants
    of i including i

    up_node_dict : {i : {anc_1 : deg_1, anc_2 : deg_2}, ...}
    or
    down_node_dict : {i : {anc_1 : deg_1, anc_2 : deg_2}, ...}

    rel_set : {r1, r2, ...}
    """

    rel_deg_dict = get_rel_deg_dict(node_dict, i)
    rel_set = {*rel_deg_dict}

    return rel_set


def get_rel_deg_dict(
    node_dict : dict[int, dict[int, int]],
    i : int,
):
    """
    If node_dict is an up_dict, get a dict mapping each ancestor
    of i (including i) to its degree from i, where the degree
    is the shortest degree over all possible paths.

    If node_dict is a down_dict, get a similar dict mapping
    each descendant of i to its degree to i

    up_node_dict : {i : {anc_1 : deg_1, anc_2 : deg_2}, ...}
    or
    down_node_dict : {i : {anc_1 : deg_1, anc_2 : deg_2}, ...}

    deg_dict : {r1 : d1, r2 : d2, ...}
    """

    deg_dict = {i : 0}
    for r1,d1 in node_dict.get(i,{}).items():

        r1_deg_dict = get_rel_deg_dict(node_dict, r1)
        for r2,d2 in r1_deg_dict.items():
            if r2 in deg_dict:
                deg_dict[r2] = min(d1+d2, deg_dict[r2])
            else:
                deg_dict[r2] = d1+d2

    return deg_dict


def get_node_paths(path_list, i):
    """
    Find all paths containing node i. Remove them
    and return two lists separately: one list with
    paths containing i and one list without.
    """
    i_paths = []
    no_i_paths = []

    for path in path_list:
        if i in path:
            i_paths.append(path)
        else:
            no_i_paths.append(path)

    return i_paths, no_i_paths


def get_anc_paths(path_list, anc_set):
    """
    Retain all unique paths in path_list that contain any ancestor
    in anc_set. Retain these paths only up to the first occurrence
    of an anc in anc_set.
    """
    anc_path_set = set()
    for path in path_list:
        semi_path = []
        for n in path:
            semi_path.append(n)
            if n in anc_set:
                anc_path_set.add(tuple(semi_path))
                break
    anc_path_list = [list(p) for p in anc_path_set]
    return anc_path_list


def get_all_paths(
    up_node_dict : dict[int, dict[int, int]],
    i : int,
    j : int,
):
    """
    Find all paths between i and j. This allows for
    consanguinity in a pedigree, which can ultimately
    allow us to compute relationships in pedigrees
    with loops.

    Args:
        up_node_dict: up node dict of the form
                    {node: {p1: d1, p2: d2},...}
                    where the number of keys in
                    each value may be 0, 1 or 2.
        i: ID of the first node
        j: ID of the second node

    Returns:
        all_path_set: Set of tuples of the form
                     {(i, ..., j), ...}
                     representing paths from i to j
        mrca_set: set of all most recent common ancestors
                  of i and j
    """

    i_path_list = get_one_way_paths(up_node_dict, i)
    j_path_list = get_one_way_paths(up_node_dict, j)

    i_anc_set = get_rel_set(up_node_dict, i)
    j_anc_set = get_rel_set(up_node_dict, j)

    common_anc_set = i_anc_set & j_anc_set

    if not common_anc_set:
        return set(),set()

    all_path_set = set()
    mrca_set = set()

    # check whether i or j is an ancestor of the other
    if i in common_anc_set: # i is an ancestor of j
        j_i_paths, j_path_list = get_node_paths(j_path_list, i)
        j_i_paths = get_anc_paths(j_i_paths, {i})
        all_path_set |= {tuple(reversed(p)) for p in j_i_paths}
        mrca_set.add(i)
    elif j in common_anc_set: # j is an ancestor of i
        i_j_paths, i_path_list = get_node_paths(i_path_list, j)
        i_j_paths = get_anc_paths(i_j_paths, {j})
        all_path_set |= {tuple(p) for p in i_j_paths}
        mrca_set.add(j)
    for i_path in i_path_list:
        for j_path in j_path_list:

            common_ancs = {*i_path} & {*j_path}
            if not common_ancs:
                continue

            i_sub_path = []
            for i_idx,n in enumerate(i_path):
                if n in common_ancs:
                    break
                i_sub_path.append(n)

            j_sub_path = []
            for j_idx,n in enumerate(j_path):
                if n in common_ancs:
                    break
                j_sub_path.append(n)

            # get the index and value of the mrca of i and j
            mrca_idx = i_idx
            mrca = i_path[mrca_idx]

            # get the path from i to j
            path = i_sub_path + [mrca] + [*reversed(j_sub_path)]

            # append path and mrca
            all_path_set.add(tuple(path))
            mrca_set.add(mrca)

    return all_path_set, mrca_set


def get_rel_stubs(
    up_dct : dict[int, dict[int, int]],
    node : int,
    gt_id_set: Optional[set[int]]=None,
):
    """
    Find all the rel tuples from node to each ID
    in gt_id_set in up_dct.

    Args:
        up_dct : up node dict of the form {id : {p1 : d1, p2 : d2},...}
                 where pi is parent i and di is the degree between id and pi.
        node: a node of interest in up_dct (can be a parent)
        gt_id_set: optionally specify a set of IDs within which to get
                   relationships.

    Returns:
        rel_stubs : dict mapping each id in gt_id_set
                    to the 3-tuple relationship tuple
                    representing its relationship to node.
    """
    if node is None:
        return {}

    # get all genotyped IDs in the pedigree (including node)
    if gt_id_set is None:
        gt_id_set = get_gt_id_set(up_dct)

    rel_stubs = {}
    for iid in gt_id_set:
        rel_tuple = get_simple_rel_tuple(
            up_node_dict=up_dct,
            i=node,
            j=iid,
        )
        rel_stubs[iid] = rel_tuple

    return rel_stubs


# @functools.lru_cache(maxsize=None)
def get_simple_rel_tuple(
    up_node_dict : dict[int, dict[int, int]],
    i : int,
    j : int,
):
    """
    From a path set, return a simple relationship
    tuple of the form (up, down, num_ancs). This kind
    of tuple only applies to outbred pedigrees. For inbred
    pedigrees, we need a different overall description of
    a relationship.

    Args:
        up_node_dict: dict of the form {node : {parent1 : deg1, parent2 : deg2},...}
        i: first node
        j: second node

    Returns:
        rel: tuple of the form (up, down, num_ancs)
    """
    if i == j:
        return (0, 0, 2)

    path_set,_ = get_all_paths(up_node_dict, i, j)

    if len(path_set) == 0:
        return None

    num_ancs = len(path_set)

    path = path_set.pop()

    up = 0
    down = 0
    for i1,i2 in get_pairs(path):
        u = 0
        d = 0
        if i1 in up_node_dict and i2 in up_node_dict[i1]:
            u = up_node_dict[i1][i2]
        elif i2 in up_node_dict and i1 in up_node_dict[i2]:
            d = up_node_dict[i2][i1]
        else:
            raise Exception("Node pair not in up_node_dict")
        up += u
        down += d

    return up, down, num_ancs


def get_rel_dict(
    up_dct : dict[int, dict[int, int]],
):
    """
    Get a dict mapping each ID in up_dct
    to each other ID to the rel tuple relating them.

    Args:
        up_dct:

    Returns:
        rel_dict: Dict of the form

            {
                i1: {
                    i2: (u,d,a),...
                },...
            }
        where (u,d,a) is the tuple
        representing the relationship
        from i1 to i2
    """
    all_id_set = get_all_id_set(up_dct)

    rel_dict = {}
    for i,j in combinations(all_id_set, r=2):
        rel = get_simple_rel_tuple(
            up_node_dict=up_dct,
            i=i,
            j=j,
        )

        if i not in rel_dict:
            rel_dict[i] = {}
        if j not in rel_dict:
            rel_dict[j] = {}

        rel_dict[i][j] = rel
        rel_dict[j][i] = reverse_rel(rel)

    for i in all_id_set:
        if i not in rel_dict:
            rel_dict[i] = {}
        rel_dict[i][i] = (0,0,2)

    return rel_dict


def shift_ids(
    ped : dict[int, dict[int, int]],
    shift : int,
):
    """
    Shift all ungenotyped IDs in ped
    so that they are smaller than min_id.

    Args:
        ped: up node dict of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node
        shift: shift all IDs in ped by shift

    Returns:
        new_ped: with all negative IDs shifted by shift
        rep_dct: dict mapping old ID to shifted ID for each
                 ID that was updated.
    """

    if shift >= 0:
        raise BonsaiError("shift must be negative.")

    all_id_set = get_all_id_set(ped)
    ug_id_set = {i for i in all_id_set if i < 0}  # ungenotyped IDs
    rep_dct = {g : g + shift for g in ug_id_set}

    new_ped = replace_ids(
        rep_dct=rep_dct,
        dct=ped,
    )

    return new_ped, rep_dct


def get_all_id_set(
    ped : dict[int, dict[int, int]],
):
    """
    Get all IDs, both genotyped and ungenotyped,
    from a node dict (up or down node dict).

    Args:
        ped: up node dict of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node

    Returns:
        all_id_set: all IDs (both positive and negative) in ped
    """
    if not ped:
        return set()
    else:
        return {*ped} | set.union(*[{*v} for v in ped.values()])


def get_min_id(
    dct : dict[int, dict[int, int]],
):
    """
    Get the minimal ID in an up (or down)
    node dict.

    Args:
        dct: up (ord down) node dict of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            or
            {node: {desc1 : deg1, desc2 : deg2,...}, ..}
            A node in a down dict can have any number of
            descendants including 0 and a node in an up dict
            can have 0, 1, or 2 parents.

    Returns:
        min_id: minimum ID in dct
    """
    all_id_set = get_all_id_set(dct)
    min_id = min(all_id_set)
    min_id = min(-1, min_id)  # ensure ID is negative
    return min_id


def replace_ids(
    rep_dct : dict[int, int],
    dct : dict[int, dict[int, int]],
):
    """
    Replace ID old_id in node dict dct with new_id
    for each key-value pair old_id:new_id in rep_dct.

    Args:
        rep_dct : Index mapping old ID to new ID.
        dct: Node dict of the form
                {node: {parent/child1 : deg1, parent/child2 : deg2}, ..}
                Can have zero, one, or two parents per node

    Returns:
        new_dct: dct with old_id replaced with new_id
    """
    if type(dct) is not dict:
        return dct

    new_dct = {
        rep_dct.get(k, k) : replace_ids(rep_dct, v)
        for k,v in dct.items()
    }

    return new_dct


def pop_node(
    node: int,
    up_dct: dict[int, dict[int, int]],
):
    """
    Pop a node off a pedigree and leave an empty ungenotyped
    node in its place. Return the popped pedigree and the
    ID of the empty node.

    Args:
        node: node to pop
        up_dct: up node dict of the form
                {node: {parent1 : deg1, parent2 : deg2}, ..}
                Can have zero, one, or two parents per node

    Returns:
        new_dct: dct with node removed
        empty_id: ID of the empty node
    """
    min_id = get_min_id(up_dct)
    empty_id = min_id - 1
    rep_dct = {node : empty_id}
    new_dct = replace_ids(rep_dct=rep_dct, dct=up_dct)
    return new_dct, empty_id


def delete_node(
    dct: dict[int, dict[int, int]],
    node: int,
):
    """
    Delete node from a node dict.

    Args:
        dct: up or down node dict of the form
            {node: {r1 : d1, r2 : d2,...}, ..}
            Can have zero, one, or two parents per node
            XOR any number of children per node.
        node: node to delete

    Returns:
        new_dct: dct with node removed. May result in
                 an unconnected graph.
    """
    new_dct = {}
    for k,v in dct.items():
        if k != node:
            new_dct[k] = {r:d for r,d in v.items() if r != node}
    return new_dct


def get_gt_id_set(
    ped : dict[int, dict[int, int]],
):
    """
    Get genotyped IDs from an up node dict.

    Args:
        ped: up node dict of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node

    Returns:
        gt_id_set: all IDs (both positive and negative) in ped
    """
    all_id_set = get_all_id_set(ped)
    gt_id_set = {i for i in all_id_set if i > 0}
    return gt_id_set


def get_partner_id_set(
    node : int,
    up_dct : dict[int, dict[int, int]],
):
    """
    Find the set of partners of node a
    in pedigree ped.

    Args:
        node: node (int)
        up_dct: up node dict of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node

    Returns:
        partner_set: set of the form {p1, p2, ...}
                     of partners of a
    """
    down_dct = reverse_node_dict(up_dct)

    child_id_set = {c for c,d in down_dct.get(node, {}).items() if d==1}

    partner_id_set = set()
    for cid in child_id_set:
        pids = {p for p,d in up_dct.get(cid, {}).items() if d==1}
        partner_id_set |= pids

    partner_id_set -= {node}

    return partner_id_set


def get_possible_connection_point_set(
    ped: Any,
) -> list[
        tuple[
            tuple[int, Optional[int], int],
            tuple[int, Optional[int], int]
        ]
    ]:
    """
    Find all possible points throgh which a pedigree (ped) can be connected
    to another pedigree. A point is a tuple of the form (id1, id2, dir),
    where id1 is the main individual through whom the pedigree is connected
    and id2 is a possible secondary connecting individual (always a partner of id1
    if they exist). id2 can be None. dir indicates whether the pedigree is
    connected up to the other pedigree or down to the other pedigree. 0=down
    1=up.

    Args:
        ped: up node dict of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node

    Returns:
        point_set: Set of the form {point1, point2,...} where
                    point i is a possible connection point of the form
                    (id1, id2, dir)
                    dir:
                        0: join down from node
                        1: join up from node
                        None: join on node
    """
    point_set = set()
    all_ids = get_all_id_set(ped)
    for a in all_ids:

        parent_to_deg = ped.get(a, {})
        if len(parent_to_deg) < 2:
            point_set.add((a, None, 1))

        partners = get_partner_id_set(a, ped)
        point_set.add((a, None, 0))
        for partner in partners:
            if (partner, a, 0) not in point_set:  # only need one orientation
                point_set.add((a, partner, 0))
            point_set.add((a, partner, None))  # try reverse orientation

        point_set.add((a, None, None))

    return point_set


def trim_empty_vals_from_dct(
    dct :  dict[int, dict[int, int]],
):
    """
    Trim empty nodes from a dict. An empty
    node is a key whose value is an empty dict
    for example, 20 in:

            {1 : {20: 1}, 20 : {}}

    Args:
        dct: up node dict or down node dict

    Returns:
        new_dct: makes a copy of dct and removes
                 empty nodes.
    """

    return {k : v for k,v in dct.items() if v != {}}


def get_founder_set(
    up_dct : dict[int, dict[int, int]],
):
    """
    Get all pedigree founders.

    Args:
        up_dct: up node dict of the form
                {node: {p1 : d1, p2 : d2}, ...}
                values may have 0, 1, or 2 keys

    Returns:
        founder_set: set of founder nodes in up_dct
    """
    up_dct = trim_empty_vals_from_dct(up_dct)
    dn_dct = reverse_node_dict(up_dct)

    founder_set = {*dn_dct} - {*up_dct}

    return founder_set


def get_subtree_node_set(
    up_dct : dict[int, dict[int, int]],
    id_set : set[int],
):
    """
    Get all nodes that are part of the subtree
    connecting all the ids in id_set.

    Args:
        up_dct: an up node dict of the form
                {node: {p1 : d1, p2 : d2}, ...}
                values may have 0, 1, or 2 keys
        id_set: A set of integer IDs

    Returns:
        node_set: set of nodes on the subtree
                  connecting the nodes in id_set
        # mrca_set: set of nodes that are most recent
        #           common ancestors of a pair of IDs
        #           in node_set.
    """
    if len(id_set) == 1:
        return id_set

    node_set = set()
    # possible_mrca_set = set()
    for i,j in combinations(id_set, r=2):
        # get all paths and mrcas between i and j
        con_path_set, mrca_set = get_all_paths(
            up_node_dict=up_dct,
            i=i,
            j=j,
        )

        # get all nodes on all paths between i and j
        for con_path in con_path_set:
            con_path_set = {*con_path}
            node_set |= con_path_set

    # WARNING: when getting the actual tree containing these nodes
    #          make sure you're getting the ancestors who are not
    #          on a connecting path if you want them. I.e., the
    #          married-in founders who are not on a path.

    return node_set  #, mrca_set


def get_sub_up_node_dict(
    up_dct: dict[int, dict[int, int]],
    id_set: set[int],
):
    """
    Get the sub up node dict corresponding to all parts
    of the tree connecting the IDs in id_set.

    Args:
        up_dct: an up node dict of the form
                {node: {p1 : d1, p2 : d2}, ...}
                values may have 0, 1, or 2 keys
        id_set: A set of integer IDs

    Returns:
        sub_up_dct: up node dict corresponding to the
                    subtree connecting the IDs in id_set
    """

    # get the nodes on the subtree connecting id_set
    subtree_node_set = get_subtree_node_set(
        up_dct=up_dct,
        id_set=id_set,
    )

    # subset the tree to these nodes
    sub_up_dct = get_sub_up_dct_for_id_set(
        up_dct=up_dct,
        id_set=subtree_node_set,
    )

    return sub_up_dct


def get_arch_founder_set(
    up_dct : dict[int, dict[int, int]],
    id_set : set[int],
):
    """
    Get all "arch" founders of a pedigree.
    These are founder nodes that are at the "peaks"
    of the pedigree diagram and which are ancestral
    to a particular node set.

    For example, below, A and E are "arch" founders for the
    id set {1,2}, whereas C is not a founder at all
    and D is a founder, but not an "arch" founder.

    However D is an arch founder for the id set {1}.

            E
           /
          /
       A B
       /  \
      /    C  D
     /      \\/
    /        \
    1         2

    Args:
        up_dct: an up node dict of the form
                {node: {p1 : d1, p2 : d2}, ...}
                values may have 0, 1, or 2 keys
        id_set: set of IDs that the founder must
            be a common ancestor of.

    Returns:
        arch_founder_set: set of all nodes that are
            arch founders.
    """

    # get the set of all pedigree founders
    founder_set = get_founder_set(up_dct)

    # get the down node dict
    dn_dct = reverse_node_dict(up_dct)

    arch_founder_set = set()
    for node in founder_set:
        desc_set = get_rel_set(
            node_dict=dn_dct,
            i=node,
        )

        if len(id_set - desc_set) == 0:
            arch_founder_set.add(node)

    return arch_founder_set


def get_best_arch_founder_set(
    arch_founder_set : set[int],
    up_dct : dict[int, dict[int, int]],
    id_set : set[int],
):
    """
    Find the arch founder(s) with the most
    descendants in id_set.

    This is used, for example, when several
    IDs in up_dct share IBD with another up_dct,
    but some of that IBD is background or from
    consanguinity and, as a result, there is
    not one single (pair of) common ancestor(s)
    of id_set in up_dct.

    We can use get_best_arch_founder_set() to find
    a common ancestor who has a bunch of descendants.
    This ancestor is more likely than another such
    ancestor to be the actual (or strongest) connection
    between the pedigrees, or to be the ancestor of
    a node that is the strongest connection between
    the pedigrees.

    Note that we probably really want the ancestor
    whose descendants share the most IBD with the
    other pedigree, but that is much more intensive
    to compute and requires passing the ibd_seg_list
    around so we do it this way instead.

    Args:
        arch_founder_set: set of IDs that are arch founders
            obtained from get_arch_founder_set()
        up_dct: an up node dict of the form
                {node: {p1 : d1, p2 : d2}, ...}
                values may have 0, 1, or 2 keys
        id_set: set of important IDs (generally
            IDs that share IBD with another pedigree).

    Returns:
        arch_set: set of arch founders with the
                  most descendants. The set will
                  have at most two IDs and if there
                  are two IDs, they will be a mate
                  pair.
    """

    # reverse node dict so we can get descendants
    dn_dct = reverse_node_dict(up_dct)

    # find the node(s) with the most descendants
    max_descs = 0
    arch_set = set()
    for node in arch_founder_set:
        # get descendants of node in id_set
        desc_set = get_rel_set(
            node_dict=dn_dct,
            i=node,
        )
        desc_set &= id_set # get descendants in id_set

        # get the number of such descendants
        num_descs = len(desc_set)

        # if num_descs is the most so far, record node
        if num_descs > max_descs:
            max_descs = num_descs
            arch_set = {node}
        elif num_descs == max_descs:
            arch_set.add(node)

    return arch_set


def get_sub_up_dct_for_id_set(
    up_dct : dict[int, dict[int, int]],
    id_set : set[int],
):
    """
    Get the sub-pedigree within up_dct
    corresponding only to nodes in node_set

    Args:
        up_dct: an up node dict of the form
                {node: {p1 : d1, p2 : d2}, ...}
                values may have 0, 1, or 2 keys
        id_set: set of all IDs in the sub pedigree

    Returns:
        sub_up_dct: subset of up_dct with
            exactly the nodes in id_set
            an up node dict of the form
                {node: {p1 : d1, p2 : d2}, ...}
            values may have 0, 1, or 2 keys
    """
    sub_up_dct = {
        i : {
            n : d
            for n,d in up_dct.get(i, {}).items()
            if n in id_set
        }
        for i in id_set
    }

    return sub_up_dct


def get_rel_ct_dict(
    dct: dict[int, dict[int, int]],
    id_set: set[int],
):
    """
    For an up_dict, get a dict mapping
    every ancestor of the IDs in id_set
    including the IDs themselves to
    the number of offspring they have
    in id_set.

    For a down_node_dict, get a dict
    mapping each descendant of a node
    in id_set to the number of ancestors
    they have in id_set.

    Args:
        dct: up_dct or down_dct
        id_set: set of integer IDs

    Returns:
        rel_ct_dict
    """
    rel_ct_dict = {}
    for iid in id_set:
        rel_set = get_rel_set(
            node_dict=dct,
            i=iid,
        )  # all ancs or descendants of i
        for r in rel_set:
            if r not in rel_ct_dict:
                rel_ct_dict[r] = 0
            rel_ct_dict[r] += 1
    return rel_ct_dict


def get_anc_to_desc_set(
    up_dct: dict[int, dict[int, int]],
    id_set: set[int],
):
    """
    Get a dict mapping each ancestor of an ID in
    id_set to the set of its descendants.

    Args:
        up_dct: an up node dict of the form
            {node: {p1 : d1, p2 : d2}, ...}
            values may have 0, 1, or 2 keys
        id_set: set of IDs in up_dct

    Returns:
        anc_to_desc_set: dict mapping each ancestor
            of an ID in id_set to the set of its descendants.
    """
    anc_set = set()
    for iid in id_set:
        anc_set |= get_rel_set(
            node_dict=up_dct,
            i=iid,
        )

    down_dct = reverse_node_dict(up_dct)
    anc_to_desc_set = {}
    for anc in anc_set:
        desc_set = get_rel_set(
            node_dict=down_dct,
            i=anc,
        )
        anc_to_desc_set[anc] = desc_set

    return anc_to_desc_set


def get_mrca_set(
    up_dct: dict[int, dict[int, int]],
    id_set: set[int],
):
    """
    Get the set of IDs that are MRCAs of the
    IDs in id_set.

    If there is no one MRCA (or pair of IDs)
    that is the MRCA of id_set, find all of the
    MRCAs that together cover id_set.

    Args:
        up_dct: an up node dict of the form
            {node: {p1 : d1, p2 : d2}, ...}
            values may have 0, 1, or 2 keys
        id_set: set of IDs in up_dct.

    Returns:
        mrca_set: set of IDs in up_dct that are
            MRCAs of id_set.
    """

    # map all ancestors of id_set to their descendants
    anc_to_desc_set = get_anc_to_desc_set(
        up_dct=up_dct,
        id_set=id_set,
    )  # includes id_set

    anc_set = {*anc_to_desc_set}

    # for each ID in anc_set, if the parent of the
    # ID has the same genotyped descendants in id_set,
    # remove the parent and all their ancestors from
    # anc_set.
    down_dct = reverse_node_dict(up_dct)
    anc_list = [*anc_set]
    for anc in anc_list:
        anc_gt_desc_set = anc_to_desc_set[anc] & id_set
        for pid in up_dct.get(anc, {}):
            pid_gt_desc_set = anc_to_desc_set[pid] & id_set
            if anc_gt_desc_set == pid_gt_desc_set:
                anc_set -= {pid}
            elif len(pid_gt_desc_set - anc_gt_desc_set) > 0:
                to_del_set = get_rel_set(
                    node_dict=down_dct,
                    i=anc,
                )
                anc_set -= to_del_set

    # remove ancestors with no unique descendants
    for anc1, anc2 in combinations(anc_set, r=2):
        anc1_gt_desc_set = anc_to_desc_set[anc1] & id_set
        anc2_gt_desc_set = anc_to_desc_set[anc2] & id_set
        # if all anc1 descendants are in anc2 descendants
        if len(anc1_gt_desc_set - anc2_gt_desc_set) == 0 and anc2_gt_desc_set != anc1_gt_desc_set:
            anc_set -= {anc1}
        # elif all anc2 descendants are in anc1 descendants
        elif len(anc2_gt_desc_set - anc1_gt_desc_set) == 0 and anc1_gt_desc_set != anc2_gt_desc_set:
            anc_set -= {anc2}

    return anc_set


def get_open_anc_set(
    up_dct: dict[int, dict[int, int]],
    node: int,
):
    """
    Find all ancestors of node including
    node itself who have at most one parent.

    Args:
        up_dct: an up node dict of the form
            {node: {p1 : d1, p2 : d2}, ...}
            values may have 0, 1, or 2 keys
        node: node in question

    Returns:
        open_anc_set: all ancestors of node
            including node with at most one
            parent.
    """
    anc_set = get_rel_set(
        node_dict=up_dct,
        i=node,
    )

    open_anc_set = {
        a
        for a in anc_set
        if a in up_dct and len(up_dct[a]) < 2
        or a not in up_dct
    }

    return open_anc_set


def get_first_open_ancestor_set(
    up_dct: dict[int, dict[int, int]],
    node: int,
):
    """
    Find the most recent ancestor(s) of node
    including node itself who have at most
    one parent.

    Args:
        up_dct: an up node dict of the form
            {node: {p1 : d1, p2 : d2}, ...}
            values may have 0, 1, or 2 keys
        node: node in question

    Returns:
        open_anc_set: set of most recent
            ancestors of node including node
            itself with at most one
            parent.
    """
    anc_deg_dict = get_rel_deg_dict(
        node_dict=up_dct,
        i=node,
    )

    # sort ancestors by degree to node
    sorted_anc_deg_list = sorted(anc_deg_dict.items(), key=lambda x: x[1])

    min_deg = 0
    open_anc_set = set()
    for anc, deg in sorted_anc_deg_list:
        pid_dict = up_dct.get(anc, {})
        num_pids = len(pid_dict)

        is_open = num_pids < 2
        is_first_entry = not open_anc_set
        has_min_deg = deg == min_deg

        if is_open and (is_first_entry or has_min_deg):
            open_anc_set.add(anc)
            min_deg = deg

    return open_anc_set


def restrict_connection_point_set(
    up_dct: dict[int, dict[int, int]],
    con_pt_set: set[int],
    id_set: set[int],
):
    """

    Restrict the set of connection points to
    include only those points on the subtree
    connecting the genotyped nodes (id_set)
    that share IBD with the other pedigee.

    Args:
        up_dct: an up node dict of the form
            {node: {p1 : d1, p2 : d2}, ...}
            values may have 0, 1, or 2 keys
        con_pt_set: output of
            get_possible_con_pt_set()
            which is a set of the form
            {(i,p,d),...}
            where i is a node, p is a possible
            partner of i, and d is 0, 1, or None.
        id_set: set of IDs in up_dct that would
            must be related once the pedigree
            is connected through a point in
            con_pt_set.

    Returns:
        rest_con_pt_set: subset of con_pt_set
            that yields a connection to every
            point in id_set.
    """

    # get the nodes on the subtree connecting id_set
    subtree_node_set = get_subtree_node_set(
        up_dct=up_dct,
        id_set=id_set,
    )

    # subset the tree to these nodes
    sub_up_dct = get_sub_up_dct_for_id_set(
        up_dct=up_dct,
        id_set=subtree_node_set,
    )

    # get the mrcas of sub_up_dct
    mrca_set = get_mrca_set(
        up_dct=sub_up_dct,
        id_set=id_set,
    )

    # get partners of the mrca set
    mrca_partner_set = set()
    for mrca in mrca_set:
        mrca_partner_set |= get_partner_id_set(
            node=mrca,
            up_dct=sub_up_dct,
        )

    # get all descendants of subtree_node_set
    down_dict = reverse_node_dict(up_dct)
    desc_set = set()
    for node in subtree_node_set:
        desc_set |= get_rel_set(
            node_dict=down_dict,
            i=node,
        )

    # get all ancestors of mrca_set and mrca_partner_set
    anc_set = set()
    for node in mrca_set | mrca_partner_set:
        anc_set |= get_rel_set(
            node_dict=up_dct,
            i=node,
        )

    # include all parents of nodes in subtree_node_set
    parent_node_set = {
        p
        for i in subtree_node_set
        for p in up_dct.get(i, {})
    }

    # get all nodes that can be a connecting point
    con_node_set = (
        subtree_node_set |
        mrca_set |
        mrca_partner_set |
        desc_set |
        anc_set |
        parent_node_set
    )

    # cycle over connection points and retain only
    # those points that have at least one node in
    # subtree_node_set.
    rest_con_pt_set = set()
    for con_pt in con_pt_set:
        i, p, d = con_pt
        if i in con_node_set or p in con_node_set:
            rest_con_pt_set.add(con_pt)

    return rest_con_pt_set


def get_likely_con_pt_set(
    up_dct: dict[int, dict[int, int]],
    id_to_shared_ibd : dict[int, float],
    rel_dict : dict[int, dict[int, Optional[tuple[int, int, int]]]],
    con_pt_set : set[int],
    max_con_pts : int=MAX_CON_PTS,
):
    """
    TODO: this would be a lot faster if we
        just computed the scaled norm difference
        between inv_deg_list and ibd_list instead
        of computing the whole peason correlation.

    Find points that are likely to be
    the connecting point from up_dct
    to another dict because their
    relative distances to the IDs
    that share IBD with the other
    pedigree are consistent with the
    amounts of IBD each of these IDs
    shares with the other pedigree.

    Args:
        up_dct: an up node dict of the form
            {node: {p1 : d1, p2 : d2}, ...}
            values may have 0, 1, or 2 keys
        id_to_shared_ibd: dict mapping each ID in up_dct
            to the total amount of IBD it shares with
            the nodes in the other pedigree.
        rel_dict: dict of the form
            {i1 : {i2 : rel,...},...}
            where rel is either None or a tuple
            of the form (up, down, num_ancs)
        con_pt_set: set of the form
                {(i,p,d),...}
            where i is a node, p is a possible
            partner of i, and d is 0, 1, or None.
            This is an optional argument. If provided,
            it will only search for valid connection
            points within this set.
        max_con_pts: maximum number of connection points
            to consider. To find these points, we find the
            top max_con_pts points with the highest
            correlation coefficient between the shared
            amounts of IBD in id_to_shared_ibd and the
            inverse of the relative degrees to the
            connection point.

    Returns:
        likely_con_pt_set: set of connetion points that
            are likely, given the relative amounts
            of IBD sharing.
    """

    # get all genotyped IDs
    gt_id_set = get_gt_id_set(up_dct)

    # if there is only one point genotyped node that shares IBD
    # then we can't use this method to distinguish among all
    # relatives of the point.
    if len([s for s in id_to_shared_ibd.values() if s > 0]) == 1:
        return con_pt_set

    # map each connection point to the
    # correlation between its inverse degrees
    # to all other nodes in the pedigree and
    # the amount of IBD each other node shares
    # with the other pedigree.
    con_pt_to_corr = {}
    for i, p, d in con_pt_set:

        # get a list of inverse degrees and a list
        # of shared IBD, both in the same order
        inv_deg_list = []
        ibd_list = []
        for gid in gt_id_set:
            # get the degrees of relationship between gid and (i,p)
            i_rel = rel_dict[gid].get(i, None)
            p_rel = rel_dict[gid].get(p, None)  # evaluates to None if p is None.
            i_deg = get_deg(i_rel)
            p_deg = get_deg(p_rel)
            # print(gid, i_rel, p_rel)

            # if connecting up&down through two common ancestors,
            # shift the degree down by 1 to reflect the two ancestors.
            if (p is not None) and (p_deg is not None) and ((d == 0) or (d is None)):
                if gid == i:
                    deg = i_deg  # is 0
                elif gid == p:
                    deg = p_deg
                elif (i_deg != INF) and (p_deg != INF):  # connected to i,p. So gid is descendant
                    deg = i_deg - 1  # assume i_deg == p_deg
                elif i_deg != INF:
                    if d == 1 and i_rel[1] > 0:  # i is connected up to other pedigree and to gid
                        deg = INF
                    else:
                        deg = i_deg
                elif p_deg != INF:
                    if d == 1 and p_rel[1] > 0:  # p is connected up to other pedigree and to gid
                        deg = INF
                    else:
                        deg = p_deg
                else:  # both i and p unrelated to gid
                    deg = INF
            elif gid == i:
                deg = i_deg  # is 0
            elif i_deg == INF:
                deg = INF
            elif d == 1 and i_rel[1] > 0:  # i is connected up to other pedigree and to gid
                deg = INF
            else:
                deg = i_deg

            ibd = id_to_shared_ibd.get(gid, 0)

            inv_deg_list.append(2**(-deg))
            ibd_list.append(ibd)

        # find the correlation between inv_deg_list and ibd_list
        corr = (
            np.dot(a=inv_deg_list, b=ibd_list) /
            (np.linalg.norm(inv_deg_list) * np.linalg.norm(ibd_list))
        )

        # record the correlation
        con_pt_to_corr[(i,p,d)] = corr

    # get the top max_con_pts points
    sorted_con_pt_to_corr = sorted(
        con_pt_to_corr.items(),
        key=lambda x: x[-1],
        reverse=True,
    )

    # get the top max_con_pts most likely (most correlated) points.
    # If k_i points have correlation c_i then return all k_i points.
    # So if the top 3 correlations have k_1, k_2, and k_3 points
    # respectively, we will return all k_1 + k_2 + k_3 points.
    corr_set = {*con_pt_to_corr.values()}
    sorted_corrs = sorted(corr_set, reverse=True)
    top_corr_set = {*sorted_corrs[:max_con_pts]}

    likely_con_pt_set = set()
    for p,c in sorted_con_pt_to_corr:
        if c in top_corr_set:
            likely_con_pt_set.add(p)

    return likely_con_pt_set


def get_common_anc_set(
    up_dct : dict[int, dict[int, int]],
    id_set : set[int],
):
    """
    Get all common ancestors of the nodes
    in id_set.

    Can be empty if nodes have no shared
    common ancestor.

    Args:
        up_dct: an up node dict of the form
                {node: {p1 : d1, p2 : d2}, ...}
                values may have 0, 1, or 2 keys
        id_set: A set of integer IDs

    Returns:
        anc_set: set of common ancestors
                 of nodes in id_set.
    """

    all_id_set = get_all_id_set(up_dct)

    anc_set = all_id_set
    for node in id_set:
        # get all paths extending up from node
        anc_path_list = get_one_way_paths(
            node_dict=up_dct,
            i=node,
        )

        # get all ancestors of node including node itself
        node_anc_set = set()
        for anc_path in anc_path_list:
            node_anc_set |= {*anc_path}

        # intersect all ancestors of node
        # with ancestors of everyone else.
        anc_set &= node_anc_set

    return anc_set


def add_parent(
    node: int,
    up_dct: dict[int, dict[int, int]],
    min_id: Optional[int]=None,
):
    """
    WARNING: modifies up_dct

    Add an ungenotyped parent to node in up_dct.

    Args:
        node: the node to add a parent to.
        up_dct: an up node dict of the form
            {node: {p1 : d1, p2 : d2}, ...}
            values may have 0, 1, or 2 keys
        min_id: optional minimal ID in up_dct.
            if min_id is not passed as an argument,
            we'll calculate it.

    Returns:
        up_dct: version of up_dct with a parent
            added for node.
        pid: ID of the parent added for node.
    """

    if node not in up_dct:
        raise BonsaiError(f"Node {node} is not in up dct.")

    pid_dict = up_dct[node]
    if len(pid_dict) >= 2:
        return up_dct, None

    if min_id is None:
        min_id = get_min_id(up_dct)

    new_pid = min_id - 1
    up_dct[node][new_pid] = 1

    return up_dct, new_pid


def fill_in_partners(
    up_dct : dict[int, dict[int, int]],
):
    """
    Fill in all missing partners of nodes
    in up_dct. This is an important pre-processing
    step to combining pedigrees because
    we can't search over possible nodes
    and partners for attaching if some
    partners aren't in the pedigree.

    Args:
        up_dct: an up node dict of the form
                {node: {p1 : d1, p2 : d2}, ...}
                values may have 0, 1, or 2 keys

    Returns:
        up_dct: with partner nodes added by
            finding every node with exactly
            one parent and filling in an
            ungenotyped parent for them.

    Does not modify up_dct
    """
    up_dct = copy.deepcopy(up_dct)
    min_id = get_min_id(up_dct)
    for node in up_dct:
        pid_dict = up_dct[node]  # parent dict
        if len(pid_dict) == 1:
            up_dct, new_pid = add_parent(
                node=node,
                up_dct=up_dct,
                min_id=min_id,
            )
            min_id = new_pid
    return up_dct


def get_split_set(up_dct, gid_set):
    """
    Cycle over all nodes in up_dct
    and get the set of genotyped IDs in gid_set
    that are descendnats of the node.

    For each node, get a frozenset containing
    the descendants and store it in split_set.

    Args:
        up_dct: an up node dict of the form
                {node: {p1 : d1, p2 : d2}, ...}
                values may have 0, 1, or 2 keys
        gid_set: set of genotyped IDs to consider

    Returns:
        split_set: set of frozensets containing
            the descendants of each meiosis.
    """
    dn_dct = reverse_node_dict(up_dct)

    split_set = set()
    for node in up_dct:
        desc_id_set = get_rel_set(
            node_dict=dn_dct,
            i=node,
        )
        desc_gid_set = frozenset(desc_id_set & gid_set)
        split_set.add(desc_gid_set)

    return split_set


def check_topo_match(up_dct1, up_dct2):
    """
    Check if the topologies of two pedigrees
    are the same.

    Args:
        up_dct1: an up node dict of the form
                {node: {p1 : d1, p2 : d2}, ...}
                values may have 0, 1, or 2 keys
        up_dct2: an up node dict of the form
                {node: {p1 : d1, p2 : d2}, ...}
                values may have 0, 1, or 2 keys

    Returns:
        match: True if the topologies are the same,
            False otherwise.
    """
    gid_set1 = get_gt_id_set(up_dct1)
    gid_set2 = get_gt_id_set(up_dct2)
    gid_set = gid_set1 & gid_set2

    splits1 = get_split_set(up_dct1, gid_set)
    splits2 = get_split_set(up_dct2, gid_set)

    match = splits1 == splits2

    return match


def delete_nodes(up_node_dict, delete_set):
    """
    Delete each node in delete_set from up_node_dict.

    Args:
        up_node_dict: dict of the form {i : {p1 : d1, p2 : d2}, ...}
        delete_set: set of nodes to delete

    Returns:
        new_up_dict: with nodes in delete_set removed.
    """
    new_up_dict = {}
    for node,info in up_node_dict.items():
        if node in delete_set:
            continue
        new_info = {p : d for p,d in info.items() if p not in delete_set}
        new_up_dict[node] = new_info
    return new_up_dict


def remove_dangly_founders(up_node_dict):
    """
    Remove founders who are not genotyped and who
    have only one child.

    Args:
        up_node_dict: dict of the form {i : {p1 : d1, p2 : d2}, ...}

    Returns:
        up_node_dict: with dangly founders removed.
    """
    while True:
        dn_node_dict = reverse_node_dict(up_node_dict)
        fid_set = {*dn_node_dict} - {*up_node_dict}
        fid_with_one_child_set = {fid for fid in fid_set if len(dn_node_dict[fid]) == 1}
        if len(fid_with_one_child_set) == 0:
            break
        up_node_dict = delete_nodes(up_node_dict, delete_set=fid_with_one_child_set)
    return up_node_dict

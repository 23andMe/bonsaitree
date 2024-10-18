import copy
from itertools import (
    combinations,
    permutations,
    product,
)
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from .constants import (
    DEG_DELTA,
    INF,
    MAX_CON_PTS,
    MAX_PEDS,
    MAX_START_PEDS,
    MAX_UP_IBD,
    MIN_SEG_LEN,
)
from .druid import (
    infer_degree_generalized_druid,
)
from .exceptions import (
    BonsaiError,
)
from .ibd import (
    get_closest_pair,
    get_closest_pair_among_sets,
    get_ibd_stats_frozenform,
    get_id_to_shared_ibd,
    get_total_ibd_between_id_sets,
)
from .likelihoods import (
    get_age_logpdf,
    get_age_mean_std_for_rel_tuple,
    get_ped_like,
)
from .pedigrees import (
    fill_in_partners,
    get_first_open_ancestor_set,
    get_gt_id_set,
    get_likely_con_pt_set,
    get_min_id,
    #get_best_mrca_set,
    get_mrca_set,
    get_possible_connection_point_set,
    get_rel_deg_dict,
    get_rel_dict,
    get_rel_stubs,
    get_simple_rel_tuple,
    remove_dangly_founders,
    replace_ids,
    restrict_connection_point_set,
    reverse_node_dict,
    shift_ids,
    trim_to_proximal,
)
from .relationships import (
    get_deg,
    get_transitive_rel,
    reverse_rel,
)


def find_closest_pedigrees(
        idx_to_id_set : dict[int, set[int]],
        ibd_seg_list : list[tuple[int, int, int, int, str, float, float, float]],
    ):
    """
    Find the two pedigrees that share the most IBD.

    Args:
        idx_to_id_set: dict mapping an integer index
                       to a set of genotyped IDs
        ibd_seg_list: List of segments of the form
                        [[id1, id2, hap1, hap2, chrom, start, end, cm]]

    Returns:
        idx1, idx2: indices of the two pedigrees that
                    share the greatest total length of
                    merged IBD
    """
    idx_list = [*idx_to_id_set]
    max_L = 0
    max_idx1 = None
    max_idx2 = None
    for idx1, idx2 in combinations(idx_list, r=2):
        id_set1 = idx_to_id_set[idx1]
        id_set2 = idx_to_id_set[idx2]
        L = get_total_ibd_between_id_sets(id_set1, id_set2, ibd_seg_list)
        if L > max_L:
            max_L = L
            max_idx1 = idx1
            max_idx2 = idx2
    return max_idx1, max_idx2


def get_deg_range(
    deg : int,
    delta : int=DEG_DELTA,
):
    """
    Get a list of all degrees between max(0, deg - delta)
    and deg+delta.

    Args:
        deg: degree
        delta: diff

    Returns:
        deg_lst : range object [min_deg, ..., max_deg] where
                  min_deg = max(0, deg-delta) and max_deg = deg+delta
    """

    min_deg = max(0, deg-delta)
    max_deg = deg+delta

    return range(min_deg, max_deg+1)


def get_tuples_set_for_deg(
    deg : int,
    dir1 : Optional[int],
    dir2 : Optional[int],
):
    """
    Get all 3-tuple relationship tuples
    that are compaitble with degree deg.

    Express returned relationship tuples
    in order from person 1 to person 2.

    Args:
        deg : int
        dir1: direction of connection from person
              1 to person 2
        dir2: direction of connection from person
              2 to person 1

    Returns:
        rel_tuple_set: set of 3-tuple relationship tuples
                       representing all relationships from
                       person 1 to person 2 that are compatible
                       with degree deg.
    """

    # if the degree is zero, we must connect "on"
    if deg == 0 and (dir1 == dir2 is None):
        return {(0,0,2)}
    elif deg == 0:
        return set()

    # if the degree is not zero, find all
    # rel tuples consistent with degree and dir1,dir2
    rel_tuple_set = set()
    if dir1 == 1 and dir2 == 1:
        for a in [1,2]:
            m = deg+a-1-2
            for m1 in range(m+1):
                m2 = m-m1
                rel_tuple = (m1+1, m2+1, a)
                if rel_tuple == (0, 0, 1):
                    pass
                rel_tuple_set.add(rel_tuple)
    elif (dir1 is None) and (dir2 is not None):
        pass
    elif (dir1 is not None) and (dir2 is None):
        pass
    elif dir1 == 0 and dir2 == 0:
        raise BonsaiError("Cannot connect down from both pairs of IDs.")
    elif dir1 == 0:
        rel_tuple_set = {(0, deg, 1)}
    elif dir2 == 0:
        rel_tuple_set = {(deg, 0, 1)}
    else:
        pass

    return rel_tuple_set


def get_rel_tuple_set(
    deg_range : npt.ArrayLike,
    dir1 : Optional[int],
    dir2 : Optional[int],
):
    """
    Get all 3-tuple relationship tuples
    that are compaitble with each degree
    in deg_range and the directions of connection
    dir1 and dir2. Express returned relationship
    tuples in order from person 1 to person 2.

    Args:
        deg_range: array-like list of degrees
        dir1: direction of connection from person
              1 to person 2
        dir2: direction of connection from person
              2 to person 1

    Returns:
        rel_tuple_set: set of 3-tuple relationship tuples
                       representing all relationships from
                       person 1 to person 2 that are compatible
                       with some degree in deg_range.
    """
    rel_tuple_set = set()
    for deg in deg_range:
        rel_tuple_set |= get_tuples_set_for_deg(
            deg=deg,
            dir1=dir1,
            dir2=dir2,
        )
    return rel_tuple_set


def get_connection_log_like(
    connection: tuple[int, int, int],
    stubs1: dict[int, Optional[tuple[int, int, int]]],
    stubs2: dict[int, Optional[tuple[int, int, int]]],
    p_stubs1: dict[int, Optional[tuple[int, int, int]]],
    p_stubs2: dict[int, Optional[tuple[int, int, int]]],
    pw_ll_cls: Any,
):
    """
    Find the composite likelihood
    of connecting two pedigrees through tuple
    "connection" connectiong node1 in pedigree 1
    to node2 in pedigree 2.

    Args:
        connection : 3-tuple relationship tuple
                     of the form (up, down, num_ancs)
        stubs1 : dict of the form {iid : tuple}
                 where tuple is the 3-tuple relationship
                 between node iid and node1.
        stubs2 : dict of the form {iid : tuple}
                 where tuple is the 3-tuple relationship
                 between node node2 and iid.
        p_stubs1 : dict of the form {iid : tuple}
                 where tuple is the 3-tuple relationship
                 between node iid and the partner of node1.
        p_stubs2 : dict of the form {iid : tuple}
                 where tuple is the 3-tuple relationship
                 between the partner of node node2 and iid.
        pw_ll_cls: instance of class with methods for getting
                   pairwise log likelihoods.
        # num_ancs: number of common ancestors

    Returns:
        log_like : log composite likelihood computed as the sum over
                   all pairwise log likelihoods of connections between
                   each ID in stubs1 and each ID in stubs2.
    """
    log_like = 0
    for rel1, stub1 in stubs1.items():
        for rel2, stub2 in stubs2.items():

            rel_list = [stub1, connection, stub2]

            if p_stubs1 and p_stubs2:  # connecting on two mate pairs
                p_stub1 = p_stubs1[rel1]
                p_stub2 = p_stubs2[rel2]
                p_rel_list = [p_stub1, connection, p_stub2]
            elif p_stubs1: # connecting on partner in pedigree 1
                p_stub1 = p_stubs1[rel1]
                p_rel_list = [p_stub1, connection, stub2]
            elif p_stubs2: # connecting on partner in pedigree 2
                p_stub2 = p_stubs2[rel2]
                p_rel_list = [stub1, connection, p_stub2]
            else:
                p_rel_list = []

            rel_tuple = get_transitive_rel(
                rel_list=rel_list,
            )
            p_rel_tuple = get_transitive_rel(
                rel_list=p_rel_list,
            )

            # account for two common ancestors
            # assume no consanguinity so that rel_tuple == p_rel_tuple
            if p_rel_tuple and rel_tuple and p_rel_tuple == rel_tuple:
                rel_tuple = (rel_tuple[0], rel_tuple[1], 2)
            elif rel_tuple:
                rel_tuple = rel_tuple  # noqa: PLW0127
            elif p_rel_tuple:
                rel_tuple = p_rel_tuple
            else:
                rel_tuple = None

            ll = pw_ll_cls.get_pw_ll(node1=rel1, node2=rel2, rel_tuple=rel_tuple)

            log_like += ll

    return log_like


def connection_is_valid(
    i1: int,
    i2: int,
    p1: int,
    p2: int,
    dir1: Optional[int],
    dir2: Optional[int],
):
    """
    Test whether a connection is valid.
    Two connections are valid if they are
    compatible with one another.

    Two connections are incompatible if
    they specify contradictory ways of
    connecting (e.g., one connection specifying
    a connection "on" while another connection
    specifies a connection "up"), or if they
    force two different genotyped individuals
    to be the same node.

    Args:
        i1: node in pedigree 1
        i2: node in pedigree 2
        p1: partner of node in pedigree 1
        p2: partner of node in pedigree 2

    Returns:
        valid: True if a valid connection, False otherwise.
    """

    # can't connect down from both node (pairs)
    if dir1 == dir2 == 0:
        return False

    # can't connect "on" one node and down/up from another
    if (dir1 is None) and (dir2 is not None):
        return False

    # can't connect "on" one node and down/up from another
    if (dir1 is not None) and (dir2 is None):
        return False

    # can't connect "on" genotyped or unspecified nodes
    if dir1 == dir2 is None:
        if (i1 != i2) and (i1 > 0) and (i2 > 0):
            return False
        if (p1 != p2):
            if p1 is None or p2 is None:
                return False
            elif  (p1 > 0) and (p2 > 0):
                return False

    return True


def restrict_rel_tuple_set_top_n(
    rel_tuple_set: set[tuple[int, int, int]],
    stubs1: dict[int, Optional[tuple[int, int, int]]],
    stubs2: dict[int, Optional[tuple[int, int, int]]],
    pw_ll_cls: Any,
    anc_id1: int,
    anc_id2: int,
    top_n: int=20,
):
    """
    TODO: We can speed up pedigree inference if we do this step
    proactively instead of filtering rel_tuples after generating
    a giant set of them. Specifically, we should only generate
    rel_tuples that are compatible with the ages of the people.

    The function get_rel_tuple_set() returns all possible
    relationship tuples between two nodes that are compatible
    with the directions dir1, dir2, and any degree in deg_range.

    However, this is many more possible relationships than are
    feasible given the ages of the people being related.

    Here, we want to restric this potentially large group of
    possible connections to onl those that are reasonably compatible
    with the ages of the people in the pedigrees being connected.

    First, we get a point estiamte of each node through which the
    connection is being made. If the node is genotyped, this age
    estimate is the age of the node. If the node is being connected
    through a genotyped partner, it is the age of the partner. If
    both the node and the partner are genotyped, it's the average
    of their ages.

    If the node is ungenotyped, the point estimate is the average
    of age estimates coming from each of the nodes in "stubs".
    These estimates are found by taking the number of generations
    difference between anc_id and the stub node, and multiplying by
    the average age gap of a generation.

    We then find the rel tuple whose numbers of generations up and
    down yields a point estimated age gap that is closest to the
    inferred age gap betwen anc_id1 and anc_id2.

    We take all rel tuples that are within, say 5 generations of
    this point estimate.

    We could find the 95% CI for such relationships or something
    like that, but let's start simple to see if this works.

    Make sure we return at least one rel tuple (the most likely one).

    Args:
        rel_tuple_set: set of all possible relationship tuples
                       between anc_id1 and anc_id2
        stubs1: dict of the form {iid : tuple}
                 where tuple is the 3-tuple relationship
                 between anc_id1 and node iid.
        stubs2: dict of the form {iid : tuple}
                 where tuple is the 3-tuple relationship
                 between anc_id2 and iid.
        pw_ll_cls: instance of class with methods for getting
                   pairwise log likelihoods.
        anc_id1: node in pedigree 1
        anc_id2: node in pedigree 2
        partner_id1: partner of anc_id1
        partner_id2: partner of anc_id2
        top_n: number of rel tuples to keep

    Returns:
        new_rel_tuple_set: set of all possible relationship tuples
                       between anc_id1 and anc_id2 that are
                       compatible with the ages of the people
                       being connected.
    """
    age_dict = pw_ll_cls.age_dict
    age1 = infer_anc_id_age(anc_id=anc_id1, stubs=stubs1, age_dict=age_dict)
    age2 = infer_anc_id_age(anc_id=anc_id2, stubs=stubs2, age_dict=age_dict)

    # cycle over each rel_tuple in rel_tuple set.
    # find the CDF of the age gap between anc_id1 and anc_id2
    # keep the rel_tuple if the CDF is > 0.05 or < 0.95
    tuple_ll_list = []
    for rel_tuple in rel_tuple_set:
        logpdf = get_age_logpdf(
            age1=age1,
            age2=age2,
            rel_tuple=rel_tuple,
        )
        tuple_ll_list.append((rel_tuple, logpdf))

    # sort the list of rel tuples by their likelihoods
    tuple_ll_list.sort(key=lambda x: x[1], reverse=True)

    # get the top n rel tuples
    new_rel_tuple_set = {*[x[0] for x in tuple_ll_list[:top_n]]}

    return new_rel_tuple_set


def infer_anc_id_age(
    anc_id: int,
    # partner_id: Optional[int],
    stubs: dict[int, Optional[tuple[int, int, int]]],
    age_dict: dict[int, float],
):
    """
    Infer the age of anc_id using the ages of its
    relatives.

    Args:
        anc_id: node in pedigree
        partner_id: Optional partner of anc_id
        stubs: dict of the form {iid : tuple}
                 where tuple is the 3-tuple relationship
                 between anc_id and node iid.
        age_dict: dict mapping node ID to age
    """
    if anc_id in age_dict:
        return age_dict[anc_id]

    # if anc_id and partner_id are both ungenotyped, infer the age
    # of anc_id from its relatives in stubs.
    age_pt_est_list = []
    for iid, rel in stubs.items():
        if rel is not None:
            mean, std = get_age_mean_std_for_rel_tuple(rel)
            rel_age = age_dict[iid]
            age_pt_est = rel_age + mean if rel_age is not None else None
            age_pt_est_list.append(age_pt_est)

    # remove None values from age_pt_est_list
    age_pt_est_list = [a for a in age_pt_est_list if a is not None]

    # if there are no "None" ages in age_pt_est_list,
    # infer the age of anc_id to be None
    if len(age_pt_est_list) == 0:
        return None

    return np.mean(age_pt_est_list)


def get_connection_degs_and_log_likes(
    point1 : tuple[int, int, int],
    point2 : tuple[int, int, int],
    up_dct1 : dict[int, dict[int, int]],
    up_dct2 : dict[int, dict[int, int]],
    pw_ll_cls: Any,
    phased_ibd_seg_list : list[
        tuple[
            int,
            int,
            int,
            int,
            str,
            float,
            float,
            float,
        ]
    ],
    condition : bool=True,
    min_seg_len : float=MIN_SEG_LEN,
    deg_range_delta : int=DEG_DELTA,
):
    """
    Use the conditioned DRUID estimator to find the degree
    relating point1 to point2 and also the likelihood of that
    connection.

    Args:
        point1: Tuple of the form (node, partner, direction)
                where node is an integer node in up_dct1, partner
                is its partner node (if it exists) or None otherwise
                and direction is
                    0: down
                    1: up
                    None: on
        point2: Tuple of the form (node, partner, direction)
                where node is an integer node in up_dct2, partner
                is its partner node (if it exists) or None otherwise
                and direction is
                    0: down
                    1: up
                    None: on
        up_dct1: up node dict of pedigree 1 of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node
        up_dct2: up node dict of pedigree 2 of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node
        pw_ll_cls: instance of class with methods for getting
                   pairwise log likelihoods.
        phased_ibd_seg_list: list of phased segments of the form
            [[id1, id2, hap1, hap2, chromosome, start, end, seg_cm]]

    Returns:
        deg1, deg2, num_common_ancs, log_like, where
            deg1 : degree up from node1 to its common ancestor
                   with node2
            deg2: degree up from node2 to its common ancestor
                   with node1
            num_common_ancs: 1 or 2
            log_like: log likelihood of the connecting lineage
                      together with the log likelihoods of the
                      pedigrees.
    """

    anc_id1, partner_id1, dir1 = point1
    anc_id2, partner_id2, dir2 = point2

    # check whether this is a valid connection
    # I.e., are the two connection types specified
    # in point1 and point2 compatible with one another?
    valid_connection = connection_is_valid(
        i1=anc_id1,
        i2=anc_id2,
        p1=partner_id1,
        p2=partner_id2,
        dir1=dir1,
        dir2=dir2,
    )

    # if not a valid connection, return an empty connection list.
    if not valid_connection:
        return []

    if dir1 is None and dir2 is None:
        est_deg = 0
        deg_range = [0]
    else:
        # TODO: memoize the following call.
        # TODO: This does not currently account for background IBD (mean_bgd_num and mean_bgd_len)
        est_deg, L_est = infer_degree_generalized_druid(
            anc_id1=anc_id1,
            anc_id2=anc_id2,
            partner_id1=partner_id1,
            partner_id2=partner_id2,
            dir1=dir1,
            dir2=dir2,
            up_dct1=up_dct1,
            up_dct2=up_dct2,
            ibd_seg_list=phased_ibd_seg_list,
            condition=condition,
            min_seg_len=MIN_SEG_LEN,
        )
        deg_range = get_deg_range(
            deg=est_deg,
            delta=deg_range_delta,
        )

    # get the relationships between each proximally-related
    # genotyped ID and anc_id1 and anc_id2.
    stubs1 = get_rel_stubs(
        up_dct=up_dct1,
        node=anc_id1,
        #gt_id_set=prox_set1,
        gt_id_set=None,
    )
    p_stubs1 = get_rel_stubs(
        up_dct=up_dct1,
        node=partner_id1,
        #gt_id_set=prox_set1,
        gt_id_set=None,
    )
    # reverse stubs1 in preparation for getting full paths between
    # proximal genotyped people in up_dict1 and proximal genotyped
    # people in up_dict2.
    rev_stubs1 = {i : reverse_rel(r) for i,r in stubs1.items()}
    rev_p_stubs1 = {i : reverse_rel(r) for i,r in p_stubs1.items()}

    stubs2 = get_rel_stubs(
        up_dct=up_dct2,
        node=anc_id2,
        #gt_id_set=prox_set2,
        gt_id_set=None,
    )
    p_stubs2 = get_rel_stubs(
        up_dct=up_dct2,
        node=partner_id2,
        #gt_id_set=prox_set2,
        gt_id_set=None,
    )

    rel_tuple_set = get_rel_tuple_set(
        deg_range=deg_range,
        dir1=dir1,
        dir2=dir2,
    )
    # restrict the rel_tuple connections in rel_tuple_set
    # to those that could reasonably connect people with the
    # ages in up_dict1 to the people with ages in up_dict2.
    rel_tuple_set = restrict_rel_tuple_set_top_n(
        rel_tuple_set=rel_tuple_set,
        stubs1=stubs1,
        stubs2=stubs2,
        pw_ll_cls=pw_ll_cls,
        anc_id1=anc_id1,
        anc_id2=anc_id2,
    )

    # cycle over all rel_tuples in rel_tuple_set and get the likelihood
    # of the resulting connection.
    log_like_list = []
    for connection in rel_tuple_set:
        log_like = get_connection_log_like(
            connection=connection,
            stubs1=rev_stubs1,
            stubs2=stubs2,
            p_stubs1=rev_p_stubs1,
            p_stubs2=p_stubs2,
            #pw_log_likes=pw_log_likes,
            pw_ll_cls=pw_ll_cls,
        )
        log_like_list.append((connection, log_like))
        if connection is None:
            import pdb
            pdb.set_trace()

    if not log_like_list:
        import pdb
        pdb.set_trace()

    return log_like_list


def get_up_only_con_pt_set(
    up_dct1: dict[int, dict[int, int]],
    up_dct2: dict[int, dict[int, int]],
    ibd_seg_list: list[tuple[int, int, int, int, str, float, float, float]],
    pw_ll_cls: Any,
    con_pt_set1: set[tuple[int, Optional[int], Optional[int]]],
    con_pt_set2: set[tuple[int, Optional[int], Optional[int]]],
    max_up_ibd: float=MAX_UP_IBD,
):
    """
    Restrict the set of connection points to only those
    which are common ancestors of the genotyped IDs
    that share IBD with the other pedigree.

    Args:
        up_dct1: up node dict of pedigree 1 of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node
        up_dct2: up node dict of pedigree 2 of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node
        ibd_seg_list: list of the form [[id1, id2, hap1, hap2, chromosome, start, end, seg_cm]]
        pw_ll_cls: instance of class with methods for computing
                   pairwise log likelihoods.
        max_up_ibd: float
            Maximum amount of IBD that two IDs can share
            in order to connect "up" only.
        con_pt_set1: set of possible connection points
                     in pedigree 1
        con_pt_set2: set of possible connection points
                     in pedigree 2

    Returns:
        con_pt_set1, con_pt_set2: sets of possible connection points
                                  in pedigree 1 and pedigree 2
                                  that are common ancestors of the
                                  genotyped IDs that share IBD with
                                  the other pedigree.
    """
    id_set1, id_set2 = get_sharing_ids(
        up_dct1=up_dct1,
        up_dct2=up_dct2,
        ibd_seg_list=ibd_seg_list,
    )

    # find out whether the two closest IDs are
    # in id_set1 and id_set2 share sufficiently
    # little IBD to apply the "connect_up_only"
    # rule. I.e., we don't want to be connecting
    # a parent and child through an "up_only" relationship.
    _, _, total_L = get_closest_pair_among_sets(
        id_set1=id_set1,
        id_set2=id_set2,
        pw_ll_cls=pw_ll_cls,
    )

    # if the closest members of the pedigrees
    # are distant, we can allow combining the pedigrees
    # through "up only." Otherwise, consider all connection
    # points.
    if total_L > max_up_ibd:
        return con_pt_set1, con_pt_set2
    else:
        # get the MRCAs of each set. If there is no
        # MRCA (e.g., due to background IBD), find
        # the MRCAs of the most IDs in the set.
        mrca_set1 = get_mrca_set(
            up_dct=up_dct1,
            id_set=id_set1,
        )
        mrca_set2 = get_mrca_set(
            up_dct=up_dct2,
            id_set=id_set2,
        )

        # get the first open ancestor(s) of each MRCA
        first_open_anc_set1 = set()
        for mrca_id in mrca_set1:
            anc_set = get_first_open_ancestor_set(
                up_dct=up_dct1,
                node=mrca_id,
            )
            first_open_anc_set1 |= anc_set
        first_open_anc_set2 = set()
        for mrca_id in mrca_set2:
            anc_set = get_first_open_ancestor_set(
                up_dct=up_dct2,
                node=mrca_id,
            )
            first_open_anc_set2 |= anc_set

        # restrict to best connecting points that can
        # connect "up"
        con_pt_set1 = {
            c
            for c in con_pt_set1
            if c[0] in first_open_anc_set1 | mrca_set1
            or c[1] in first_open_anc_set1 | mrca_set1
        }
        con_pt_set2 = {
            c
            for c in con_pt_set2
            if c[0] in first_open_anc_set2 | mrca_set2
            or c[1] in first_open_anc_set2 | mrca_set2
        }
    return con_pt_set1, con_pt_set2


def get_restricted_connection_point_sets(
    up_dct1: dict[int, dict[int, int]],
    up_dct2: dict[int, dict[int, int]],
    ibd_seg_list: list[tuple[int, int, int, int, str, float, float, float]],
    con_pt_set1: set[tuple[int, Optional[int], Optional[int]]],
    con_pt_set2: set[tuple[int, Optional[int], Optional[int]]],
):
    """
    Restrict the set of connection points to
    include only those points on the subtree
    connecting the genotyped nodes (id_set)
    that share IBD with the other pedigee.

    Args:
        up_dct1: up node dict of pedigree 1 of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node
        up_dct2: up node dict of pedigree 2 of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node
        ibd_seg_list: list of the form [[id1, id2, hap1, hap2, chromosome, start, end, seg_cm]]
        con_pt_set1: set of possible connection points
                     in pedigree 1
        con_pt_set2: set of possible connection points
                     in pedigree 2
    """
    id_set1, id_set2 = get_sharing_ids(
        up_dct1=up_dct1,
        up_dct2=up_dct2,
        ibd_seg_list=ibd_seg_list,
    )
    con_pt_set1 = restrict_connection_point_set(
        up_dct=up_dct1,
        con_pt_set=con_pt_set1,
        id_set=id_set1,
    )
    con_pt_set2 = restrict_connection_point_set(
        up_dct=up_dct2,
        con_pt_set=con_pt_set2,
        id_set=id_set2,
    )
    return con_pt_set1, con_pt_set2


def get_max_con_pt_sets(
    up_dct1: dict[int, dict[int, int]],
    up_dct2: dict[int, dict[int, int]],
    ibd_seg_list: list[tuple[int, int, int, int, str, float, float, float]],
    con_pt_set1: set[tuple[int, Optional[int], Optional[int]]],
    con_pt_set2: set[tuple[int, Optional[int], Optional[int]]],
    max_con_pts: int=MAX_CON_PTS,
):
    """
    Find the set of max_con_pts most likely connection points
    that connect two pedigrees. This function looks for points
    such that the degree from the point to all genotyped IDs
    is correlated with the amount of IBD shared between the ID
    and the other pedigree.

    Args:
        up_dct1: up node dict of pedigree 1 of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node
        up_dct2: up node dict of pedigree 2 of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node
        ibd_seg_list: list of the form [[id1, id2, hap1, hap2, chromosome, start, end, seg_cm]]
        max_con_pts: maximum number of points through
                     which to try connecting a pair of pedigrees.
    """
    # find IDs in pedigree 1 that share IBD with IDs in
    # pedigree 2 and vice versa
    id_set1, id_set2 = get_sharing_ids(
        up_dct1=up_dct1,
        up_dct2=up_dct2,
        ibd_seg_list=ibd_seg_list,
    )

    # get dicts mapping each ID in upd_dct1 to the
    # total IBD it shares with all IDs in upd_dct2
    id_to_shared_ibd1 = get_id_to_shared_ibd(
        id_set1=id_set1,
        id_set2=id_set2,
        ibd_seg_list=ibd_seg_list,
    )
    id_to_shared_ibd2 = get_id_to_shared_ibd(
        id_set1=id_set2,
        id_set2=id_set1,
        ibd_seg_list=ibd_seg_list,
    )

    # get rel dicts for each up_dct
    rel_dict1 = get_rel_dict(up_dct1)
    rel_dict2 = get_rel_dict(up_dct2)

    # restrict the connecting points to only the top few
    # that have a good chance of being the correct ones
    con_pt_set1 = get_likely_con_pt_set(
        up_dct=up_dct1,
        id_to_shared_ibd=id_to_shared_ibd1,
        rel_dict=rel_dict1,
        con_pt_set=con_pt_set1,
        max_con_pts=max_con_pts,
    )
    con_pt_set2 = get_likely_con_pt_set(
        up_dct=up_dct2,
        id_to_shared_ibd=id_to_shared_ibd2,
        rel_dict=rel_dict2,
        con_pt_set=con_pt_set2,
        max_con_pts=max_con_pts,
    )

    return con_pt_set1, con_pt_set2


def get_connecting_points_degs_and_log_likes(
    up_dct1: Any,
    up_dct2: Any,
    pw_ll_cls: Any,
    ibd_seg_list : list[tuple[int, int, int, int, str, float, float, float]],
    condition : bool=False,
    min_seg_len : float=MIN_SEG_LEN,
    max_con_pts : int=INF,
    restrict_connection_points : bool=False,
    connect_up_only : bool=False,
) -> list[list[Any]]:
    """
    Cycle over all nodes. Make node dicts connecting up_dct1 and up_dct2 through
    all possible common ancestors with all possible degrees.
    Return the set of most likey ancestor pairs and their degrees and log likelihoods,
    where likelihoods are the gamma likelihoods computed
    between clades and do not include the full composite likelihoods.
        [(ca1,ca2,deg,log_like),(ca1,ca2,deg,log_like),...]

    Args:
        up_dct1: up node dict of pedigree 1 of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node
        up_dct2: up node dict of pedigree 2 of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node
        pw_ll_cls: instance of class with methods for computing
                   pairwise log likelihoods.
        ibd_seg_list: list
            List of the form
                [[id1, id2, hap1, hap2, chromosome, start, end, seg_cm]]
        condition: bool
            Whether or not to condition on observing at least one
            segment when computing the DRUID estimate. Currently not
            implemented.
        min_seg_len: float
            Minimum observable segment length.
        max_con_pts: maximum number of points through which
            to try connecting a pair of pedigrees.
        restrict_connection_points: bool
            Whether or not to restrict the connection points
            through which the pedigrees can be connected using
            the approach implemented in pedigrees.restrict_connection_point_set().
        connect_up_only: when connecting two pedigrees that are
            sufficiently far apart (minimum shared IBD between
            their two closest IDs), use an
            approach similar to Bonsai v2. In other words,
            only
            look for connection points in a pedigree that are
            common ancestors of the genotyped nodes that share
            IBD with the other pedigree (id_set). If backround IBD or
            cryptic inbreeding causes id_set to have no common
            ancestor, use the ancestor with the most descendants
            in id_set. Also use the most recent such ancestor
            who has at most one parent (open ancestor).

    Returns:
        anc_deg_log_like_list: list
            List of the form
                [[point1, point2, deg1, deg2, num_common_ancs, log_like],...]
            ordered from most to least likely.
    """

    # get all possible connecting points
    con_pt_set1 = get_possible_connection_point_set(up_dct1)
    con_pt_set2 = get_possible_connection_point_set(up_dct2)

    # BEGIN filtering block =========================================================
    if connect_up_only:
        con_pt_set1, con_pt_set2 = get_up_only_con_pt_set(
            up_dct1=up_dct1,
            up_dct2=up_dct2,
            ibd_seg_list=ibd_seg_list,
            pw_ll_cls=pw_ll_cls,
            con_pt_set1=con_pt_set1,
            con_pt_set2=con_pt_set2,
        )
    if restrict_connection_points:
        con_pt_set1, con_pt_set2 = get_restricted_connection_point_sets(
            up_dct1=up_dct1,
            up_dct2=up_dct2,
            ibd_seg_list=ibd_seg_list,
            con_pt_set1=con_pt_set1,
            con_pt_set2=con_pt_set2,
        )
    if max_con_pts < INF:
        con_pt_set1, con_pt_set2 = get_max_con_pt_sets(
            up_dct1=up_dct1,
            up_dct2=up_dct2,
            ibd_seg_list=ibd_seg_list,
            con_pt_set1=con_pt_set1,
            con_pt_set2=con_pt_set2,
            max_con_pts=max_con_pts,
        )
    # END filtering block =========================================================

    point_connection_log_like_list = []
    for point1 in con_pt_set1:
        for point2 in con_pt_set2:
            connection_log_like_list = get_connection_degs_and_log_likes(
                point1=point1,
                point2=point2,
                up_dct1=up_dct1,
                up_dct2=up_dct2,
                pw_ll_cls=pw_ll_cls,
                phased_ibd_seg_list=ibd_seg_list,
                condition=condition,
                min_seg_len=MIN_SEG_LEN,
            )
            info_list = []
            for con, log_like in connection_log_like_list:
                info_list.append([point1, point2, con, log_like])
            point_connection_log_like_list += info_list

    # filter duplicate connections
    # TODO: This may no longer be necessary since we have
    #       re-worked the way that the number of ancestors
    #       is specified (we don't iterate over it anymore)
    #       and I think that was causing the duplicates.
    point_connection_log_like_list = filter_dup_cons(
        point_connection_log_like_list
    )

    # sort from most to least likely, and break likelihood ties
    # by sorting from shortest to longest degree
    point_connection_log_like_list = sorted(
        point_connection_log_like_list,
        key = lambda x: (x[-1], -get_deg(x[-2])),
        reverse=True,
    )

    return point_connection_log_like_list


def filter_dup_cons(con_list):
    """
    Some connections may be duplicates.

    I think this is hard to filter duplicates in a
    proactive way.

    Duplicates come from the fact that we cyle over
    a in [1, 2] in get_connecting_points_degs_and_log_likes()
    above. Within that loop, we
    cycle over all connections between point1 and point2.
    Some of these connections imply a number of ancestors
    a. Others don't. It's a little tricky to filter these
    connections ahead of time so that resulting connection
    has a ancestors. It is easier to filter after the fact.
    """
    con_tuple_set = {tuple(c) for c in con_list}
    return [*con_tuple_set]


def connect_pedigrees_through_points(
    id1 : int,
    id2 : int,
    pid1 : Optional[int],
    pid2 : Optional[int],
    up_dct1 : dict[int, dict[int, int]],
    up_dct2 : dict[int, dict[int, int]],
    deg1 : int,
    deg2 : int,
    num_ancs : int,
    simple : bool=True,
):
    """
    Connect up_dct1 to up_dct2 through points id1 in up_dct1
    and id2 in up_dct2. Also connect through partner points
    pid1 and pid2, if indicated. Connect id1 to id2 through
    a relationship specified by (deg1, deg2, num_ancs).

    Assumes that the ID sets of the two pedigrees are disjoint.

    Args:
        id1: node in pedigree 1
        id2: node in pdiegree 2
        pid1: optional partner of id1
        pid2: optional partner of id2
        up_dct1: up node dict representing pedigree 1
        up_dct2: up node dict representing pedigree 2
        deg1: connect through relationship (deg1, deg2, num_ancs)
        deg2: connect through relationship (deg1, deg2, num_ancs)
        num_ancs: connect through relationship (deg1, deg2, num_ancs)
        simple: boolean determining how the pedigrees are connected.
                 if simple == True, then we only allow id1 to match id2
                 and pid1 to match pid2 when joining the pedigrees.
                 If simple == False, then we look at all ways of overlapping
                 the pedigrees such that id1->id2 and pid1->pid2.

    Returns:
        up_dct: up node dict representing up_dct1 connected to
                up_dct2 through (deg1, deg2, num_ancs)
    """

    # can't connect "on" genotyped nodes
    if deg1 == deg2 == 0 and (id1 > 0 and id2 > 0) and id1 != id2:
        return []

    # can't connect "on" genotyped or non-existent partner nodes
    if deg1 == deg2 == 0 and (pid1 != pid2):
        if pid1 is None or pid1 is None:
            return []
        elif pid1 > 0 and pid2 > 0:
            return []

    up_dct1 = copy.deepcopy(up_dct1)
    up_dct2 = copy.deepcopy(up_dct2)

    if deg1 > 0:
        up_dct1, _, id1, pid1 = extend_up(
            iid=id1,
            deg=deg1,
            num_ancs=num_ancs,
            up_dct=up_dct1,
        )

    if deg2 > 0:
        up_dct2, _, id2, pid2 = extend_up(
            iid=id2,
            deg=deg2,
            num_ancs=num_ancs,
            up_dct=up_dct2,
        )

    # shift IDs so that they don't overlap
    min_id = get_min_id(up_dct1)-1
    up_dct2, id_map = shift_ids(
        ped=up_dct2,
        shift=min_id,
    )
    id2 = id_map.get(id2, id2)
    pid2 = id_map.get(pid2, pid2)

    # get a mapping of IDs in up_dct1 to match
    # with IDs in up_dct2
    if simple:
        if (pid1 is not None) and (pid2 is not None):
            id_map_list = [
                {id1 : id2, pid1 : pid2}
            ]
        else:
            id_map_list = [
                {id1 : id2}
            ]
    else:
        id_map_list = get_all_matches(
            id1=id1,
            id2=id2,
            pid1=pid1,
            pid2=pid2,
            up_dct1=up_dct1,
            up_dct2=up_dct2,
        )

    # connect up_dct1 to up_dct2 in all
    # ways specified in id_map_list
    connect_dct_list = []
    for id_map in id_map_list:

        up_dct = connect_on(
            id_map=id_map,
            up_dct1=up_dct1,
            up_dct2=up_dct2,
        )
        connect_dct_list.append(up_dct)

    return connect_dct_list


def extend_up(
    iid: int,
    deg: int,
    num_ancs: int,
    up_dct: dict[int, dict[int, int]],
):
    """
    Extend a lineage up from iid in up node dict
    up_dct. Does not change up_dct. Returns a copy.

    Args:
        iid: it to extend up from
        deg: number of degrees up to ancestor(s)
        num_ancs: number of ancestors
        up_dct: up node dict in which iid is found

    Returns:
        ext_up_dict: extended version of up_dct
        prev_id: the ID that was most recently extended from
        new_id: the final ID that was added in the upward chain
        part_id: partner of new_id if num_ancs == 2
    """
    up_dct = copy.deepcopy(up_dct)

    if deg == 0:
        return up_dct, None, iid, None

    if deg == 1:
        if iid in up_dct and (len(up_dct[iid]) + num_ancs > 2):
            raise BonsaiError(
                f"""ID {iid} cannot be extended by degree 1 with
                {num_ancs} ancestors. It would yield >2 parents."""
            )

    if deg > 1:
        if iid in up_dct and len(up_dct[iid])==2:
            raise BonsaiError(f"ID {iid} cannot be extended. It would yield >2 parents.")

    min_id = get_min_id(up_dct)
    new_id = min(-1, min_id-1)

    prev_id = None
    part_id = None
    while deg > 0:

        if iid not in up_dct:
            up_dct[iid] = {}

        if len(up_dct[iid]) >= 2:
            raise BonsaiError(
                f"Attempting to add parent to {iid} in pedigree {up_dct}."
            )

        up_dct[iid][new_id] = 1

        # add the partner if num_ancs == 2
        if deg == 1 and num_ancs == 2:
            part_id = new_id-1
            up_dct[iid][part_id] = 1

        # update IDs
        prev_id = iid
        if deg > 1:
            iid = new_id
            new_id -= 1

        deg -= 1

    return up_dct, prev_id, new_id, part_id


def get_all_up_matches(
    id1 : int,
    id2 : int,
    up_dct1 : dict[int, dict[int, int]],
    up_dct2 : dict[int, dict[int, int]],
):
    """
    If id1 in up_dct1 is matched with id2 in
    up_dct2, find all ways in which the ancestors
    of id1 can be matched with the ancestors of id2.

    Args:
        id1 : ID in up_dct1 that matches id2 in up_dct2
        id2 : ID in up_dct2 that matches id1 in up_dct1
        up_dct1 : dict mapping each person to their parental
                  nodes. Has the form
                    {node: {parent1 : deg1, parent2 : deg2}, ..}
                    Can have zero, one, or two parents per node
        up_dct2 : dict mapping each person to their parental
                  nodes. Has the form
                    {node: {parent1 : deg1, parent2 : deg2}, ..}
                    Can have zero, one, or two parents per node

    Returns:
        lst : list of dicts mapping nodes in up_dct1 to
              nodes in up_dct2.
    """

    id_map = {id1 : id2}

    if id1 > 0 or id2 > 0:
        return []

    if (id1 not in up_dct1) or (id2 not in up_dct2):
        return [id_map]

    # get parents of id1 and id2
    pids1 = [*up_dct1[id1]]
    pids2 = [*up_dct2[id2]]
    if len(pids1) == 2 and len(pids2) == 2:

        lst = []

        lst1 = get_all_up_matches(
            id1=pids1[0],
            id2=pids2[0],
            up_dct1=up_dct1,
            up_dct2=up_dct2,
        )
        lst2 = get_all_up_matches(
            id1=pids1[1],
            id2=pids2[1],
            up_dct1=up_dct1,
            up_dct2=up_dct2,
        )

        for m1 in lst1:
            for m2 in lst2:
                new_map = {**id_map, **m1, **m2}
                lst.append(new_map)

        lst1 = get_all_up_matches(
            id1=pids1[1],
            id2=pids2[0],
            up_dct1=up_dct1,
            up_dct2=up_dct2,
        )
        lst2 = get_all_up_matches(
            id1=pids1[0],
            id2=pids2[1],
            up_dct1=up_dct1,
            up_dct2=up_dct2,
        )

        for m1 in lst1:
            for m2 in lst2:
                new_map = {**id_map, **m1, **m2}
                lst.append(new_map)

    elif len(pids1) == 1 and len(pids2) == 2:

        lst = get_all_up_matches(
            id1=pids1[0],
            id2=pids2[0],
            up_dct1=up_dct1,
            up_dct2=up_dct2,
        )
        lst += get_all_up_matches(
            id1=pids1[0],
            id2=pids2[1],
            up_dct1=up_dct1,
            up_dct2=up_dct2,
        )

        for m in lst:
            new_map = {**id_map, **m}
            lst.append(new_map)

    elif len(pids1) == 2 and len(pids2) == 1:

        lst = get_all_up_matches(
            id1=pids1[0],
            id2=pids2[0],
            up_dct1=up_dct1,
            up_dct2=up_dct2,
        )
        lst += get_all_up_matches(
            id1=pids1[1],
            id2=pids2[0],
            up_dct1=up_dct1,
            up_dct2=up_dct2,
        )

        for m in lst:
            new_map = {**id_map, **m}
            lst.append(new_map)

    elif len(pids1) == 1 and len(pids2) == 1:

        lst = get_all_up_matches(
            id1=pids1[0],
            id2=pids2[0],
            up_dct1=up_dct1,
            up_dct2=up_dct2,
        )

        for m in lst:
            new_map = {**id_map, **m}
            lst.append(new_map)

    return lst


def get_all_down_matches(
    id1 : int,
    id2 : int,
    down_dct1 : dict[int, dict[int, int]],
    down_dct2 : dict[int, dict[int, int]],
):
    """
    Find all ways of matching nodes in
    down_dct1 with nodees in down_dct2
    that are compatible with matching id1
    in down_dct1 with id2 in down_dct2.

    This is the equivalent of get_all_up_matches()
    for the case of matching descendants
    using down node dicts.

    Args:
        id1 : ID in up_dct1 that matches id2 in up_dct2
        id2 : ID in up_dct2 that matches id1 in up_dct1
        down_dct1 : dict mapping each person to their child
                    nodes. Has the form
                    {node: {child1 : deg1, child2 : deg2}, ..}
                    Can have zero, one, or two parents per node
        down_dct2 : dict mapping each person to their child
                    nodes. Has the form
                    {node: {child1 : deg1, child2 : deg2}, ..}
                    Can have zero, one, or two parents per node

    Returns:
        lst : list of dicts mapping nodes in down_dct1 to
              nodes in down_dct2.
    """
    id_map = {id1 : id2}

    if id1 > 0 or id2 > 0:
        return []

    if (id1 not in down_dct1) or (id2 not in down_dct2):
        return [id_map]

    # get children of id1 and id2
    all_cids1 = [*down_dct1[id1]]
    all_cids2 = [*down_dct2[id2]]

    # get the fewest number of children of id1 or id2
    num_cids1 = len(all_cids1)
    num_cids2 = len(all_cids2)
    min_num_ids = min(num_cids1, num_cids2)

    # cycle over all possible ways of matching children
    lst = [id_map]
    for r in range(1, min_num_ids+1):
        for cids1 in combinations(all_cids1, r=r):
            for cids2 in combinations(all_cids2, r=r):
                 for cids1_perm in permutations(cids1):
                    lsts = []
                    for c1,c2 in zip(cids1_perm, cids2):
                        match_list = get_all_down_matches(
                            id1=c1,
                            id2=c2,
                            down_dct1=down_dct1,
                            down_dct2=down_dct2,
                        )
                        lsts.append(match_list)
                    for prod in product(*lsts):
                        mp = {}
                        for p in prod:
                            mp.update(p)
                        mp.update(id_map)
                        lst.append(mp)

    return lst


def get_all_matches(
    id1 : int,
    id2 : int,
    pid1 : Optional[int],
    pid2 : Optional[int],
    up_dct1 : dict[int, dict[int, int]],
    up_dct2 : dict[int, dict[int, int]],
):
    """
    If id1 in up_dct1 is matched with id2 in
    up_dct2, find all ways in which the ancestors
    and the descedants of id1 can be matched with
    the ancestors and descendants of id2.

    Args:
        id1 : ID in up_dct1 that matches id2 in up_dct2
        id2 : ID in up_dct2 that matches id1 in up_dct1
        pid1 : ID of partner of id1
        pid2 : ID of partner of id2
        up_dct1 : dict mapping each person to their parental
                  nodes. Has the form
                    {node: {parent1 : deg1, parent2 : deg2}, ..}
                    Can have zero, one, or two parents per node
        up_dct2 : dict mapping each person to their parental
                  nodes. Has the form
                    {node: {parent1 : deg1, parent2 : deg2}, ..}
                    Can have zero, one, or two parents per node

    Returns:
        all_match_list : list of dicts mapping nodes in up_dct1 to
                         nodes in up_dct2.
    """

    down_dct1 = reverse_node_dict(up_dct1)
    down_dct2 = reverse_node_dict(up_dct2)

    # get ancestral matches for id1 and id2
    up_match_list = get_all_up_matches(
        id1=id1,
        id2=id2,
        up_dct1=up_dct1,
        up_dct2=up_dct2,
    )

    # get ancestral matches for pid1 and pid2
    p_up_match_list = [{}]
    if (pid1 is not None) and (pid2 is not None):
        p_up_match_list = get_all_up_matches(
            id1=pid1,
            id2=pid2,
            up_dct1=up_dct1,
            up_dct2=up_dct2,
        )

    # get descendant matches for id1 and id2
    down_match_list = get_all_down_matches(
        id1=id1,
        id2=id2,
        down_dct1=down_dct1,
        down_dct2=down_dct2,
    )

    all_match_list = []
    for up_matches in up_match_list:
        for p_up_matches in p_up_match_list:
            for down_matches in down_match_list:
                matches = {**up_matches, **p_up_matches, **down_matches}
                all_match_list.append(matches)

    return all_match_list


def connect_on(
    id_map : dict[Optional[int], Optional[int]],
    up_dct1 : dict[int, dict[int, int]],
    up_dct2 : dict[int, dict[int, int]],
):
    """
    Join up node dict up_dct1 to up node dict
    up_dct2 on the points specified in id_map.

    Args:
        id_map: dict mapping a subset of IDs
                in up_dct1 to the corresponding IDs in up_dct2.
        up_dct1 : dict mapping each person to their parental
                  nodes. Has the form
                    {node: {parent1 : deg1, parent2 : deg2}, ..}
                    Can have zero, one, or two parents per node
        up_dct2 : dict mapping each person to their parental
                  nodes. Has the form
                    {node: {parent1 : deg1, parent2 : deg2}, ..}
                    Can have zero, one, or two parents per node

    Returns:
        up_dct: combined up_dct1 with up_dct2 accomplished
                by matching keys of id_map with the corresponding
                values.
    """

    up_dct1 = copy.deepcopy(up_dct1)
    up_dct2 = copy.deepcopy(up_dct2)

    # resolve IDs between pedigrees. I.e.,
    # map each ID k in up_dct1 and each ID v in
    # up_dct2 to a common ID, p.
    id_map1 = {}
    id_map2 = {}
    for k,v in id_map.items():
        if k > 0:
            p = k
        elif v > 0:
            p = v
        else:
            p = k
        id_map1[k] = p
        id_map2[v] = p

    # replace IDs
    up_dct1 = replace_ids(
        rep_dct=id_map1,
        dct=up_dct1,
    )
    up_dct2 = replace_ids(
        rep_dct=id_map2,
        dct=up_dct2,
    )

    for k,v2 in up_dct2.items():
        if k not in up_dct1:
            up_dct1[k] = v2
        else:
            v1 = up_dct1[k]
            v1.update(v2)

            if len(v1) <= 2:
                up_dct1[k] = v1
            else:
                return {}

    return up_dct1


def combine_pedigrees(
    up_dct1: Any,
    up_dct2: Any,
    pw_ll_cls: Any,
    ibd_seg_list: list[tuple[int, int, int, int, str, float, float, float]],
    condition : bool = True,
    max_peds: int = MAX_PEDS,
    max_con_pts: int = INF,
    min_seg_len : float=MIN_SEG_LEN,
    restrict_connection_points : bool=False,
    connect_up_only: bool=False,
    fig_dir: Optional[str]=None,
) -> list[Any]:
    """
    Combine pedigrees up_dct1 and up_dct2 through the top most likely
    degrees and common ancestor pairs

    Args:
        up_dct1: up node dict for pedigree 1. Dict of the form
                {node: {parent1 : deg1, parent2 : deg2}, ..}
        up_dct2: up node dict for pedigree 2. Dict of the form
                {node: {parent1 : deg1, parent2 : deg2}, ..}
        pw_ll_cls:
            instance of class with methods for getting pairwise log
            likelihoods.
        ibd_seg_list: list of phased segments.
        condition: bool
            condition on observing any IBD when computing the DRUID estimate.
        max_peds: Maximum number of pedigrees to build.
        min_seg_len: minimum observable segment length in cM
        connect_up_only: when connecting two pedigrees that are
            sufficiently far apart (minimum shared IBD between
            their two closest IDs), use an
            approach similar to Bonsai v2. In other words,
            only
            look for connection points in a pedigree that are
            common ancestors of the genotyped nodes that share
            IBD with the other pedigree (id_set). If backround IBD or
            cryptic inbreeding causes id_set to have no common
            ancestor, use the ancestor with the most descendants
            in id_set. Also use the most recent such ancestor
            who has at most one parent (open ancestor).
        fig_dir: directory to save figures to for debugging.

    Returns:
        ped_like_list: List of the form [[ped, log_like], ..]
                       showing the top max_peds pedigrees that
                       were built by combining up_dct1 and up_dct2
                       and their log likelihoods.
                       ped is an up node dict of the form
                       {node: {parent1 : deg1, parent2 : deg2}, ..}
    """

    # get a list of the form [[point1, point2, deg1, deg2, num_common_ancs, log_like], ...]
    anc_deg_log_like_list = get_connecting_points_degs_and_log_likes(
        up_dct1=up_dct1,
        up_dct2=up_dct2,
        pw_ll_cls=pw_ll_cls,
        ibd_seg_list=ibd_seg_list,
        condition=condition,
        min_seg_len=min_seg_len,
        max_con_pts=max_con_pts,
        restrict_connection_points=restrict_connection_points,
        connect_up_only=connect_up_only,
    )

    ped_log_like_list : list[Any] = list()
    ped_idx = 0
    while (len(ped_log_like_list) < max_peds) and (ped_idx < len(anc_deg_log_like_list)):
        (
            point1,
            point2,
            rel_tuple,
            log_like,
        ) = anc_deg_log_like_list[ped_idx]

        # don't explore impossible pedigrees
        if log_like <= -INF:
            break

        up_dct1_copy = copy.deepcopy(up_dct1)
        up_dct2_copy = copy.deepcopy(up_dct2)

        id1, pid1, dir1 = point1
        id2, pid2, dir2 = point2

        deg1, deg2, num_ancs = rel_tuple

        # we must always add in the relevant
        # partner when connecting directly up/down
        if deg1 == 0 or deg2 == 0:
            num_ancs = 2

        try:
            new_ped_list = connect_pedigrees_through_points(
                id1=id1,
                id2=id2,
                pid1=pid1,
                pid2=pid2,
                up_dct1=up_dct1_copy,
                up_dct2=up_dct2_copy,
                deg1=deg1,
                deg2=deg2,
                num_ancs=num_ancs,
                simple=True,
            )
        except BonsaiError:
            ped_idx += 1
            continue

        if new_ped_list == []:
            ped_idx += 1
            continue

        new_ped = new_ped_list[0]
        if new_ped == {}:
            ped_idx += 1
            continue

        new_ped_log_like = get_ped_like(new_ped, pw_ll_cls)
        ped_log_like_list.append((new_ped, new_ped_log_like))

        ped_idx += 1

    ped_log_like_list = sorted(ped_log_like_list, key=lambda x: -x[-1])

    return ped_log_like_list


def merge_up_dct_ll_lists(
    up_dct_ll_list1 : list[tuple[dict[int, dict[int, int]], float]],
    up_dct_ll_list2 : list[tuple[dict[int, dict[int, int]], float]],
):
    """
    Take two up_dict log like lists and merge
    them into a single list.

    Each list is of the form [[up_dct, ll], ...]
    where up_dct is an up node dict of a pedigree
    and ll is the log likelihood of the pedigree.

    Return a list of the form [[up_dct1, up_dct2, ll1+ll2], ...]
    where up_dct1 is from up_dct_ll_list1 and up_dct2 is from
    up_dct_ll_list2.

    Args:
        up_dct_ll_list1 : List of the form [[up_dct, ll],...]
        up_dct_ll_list2 : List of the form [[up_dct, ll],...]

    Retuns:
        up_dct_ll_list : list of the form [[up_dct1, up_dct2, ll1+ll2], ...]
        sorted from most to least likely
    """
    up_dct_ll_list = []
    for u1,l1 in up_dct_ll_list1:
        for u2,l2 in up_dct_ll_list2:
            entry = [u1, u2, l1 + l2]
            up_dct_ll_list.append(entry)

    return sorted(up_dct_ll_list, key=lambda x: x[-1], reverse=True)


def combine_up_dicts(  # noqa PLR0915
    idx_to_up_dict_ll_list : dict[int, list[tuple[dict[int, dict[int, int]], float]]],
    id_to_idx : dict[int, int],
    idx_to_id_set : dict[int, set[int]],
    ibd_seg_list: list[tuple[int, int, int, int, str, float, float, float]],
    pw_ll_cls: Any,
    condition : bool = True,
    max_peds: int = MAX_PEDS,
    max_start_peds: int = MAX_START_PEDS,
    max_con_pts: int = INF,
    min_seg_len : float=MIN_SEG_LEN,
    restrict_connection_points : bool=False,
    connect_up_only : bool=False,
    db_fig_base_dir: Optional[str]=None,
    true_ped: Optional[dict[int, dict[int, int]]]=None,
):
    """"
    Take in a list of up node dicts (can be one person per dict
    if we are assembling small pedigrees).

    Find the two pedigrees with the closest IBD sharing
    among any pair of individuals (one from the first pedigree
    and the other from the second pedigree).

    Connect those pedigrees in all top ways.

    Args:
        idx_to_up_dict_ll_list: Dict of the form
            {
                idx: [[up_dct1, ll1], ...],...
            }
            Mapping a unique index for each pedigree to
            a list of possible versions of the pedigree
            and their log likelihoods.

            For each idx, the list of pedigrees is ordered
            from most to least likely.

        id_to_idx: dict of the form {id: idx,...} mapping
                   each genotype ID to the index of the
                   pedigree in which it is placed.
        idx_to_id_set: dict of the form {idx : {i1, i2, ...}, ...}
                       mapping each pedigree index to the
                       set of genotype IDs in the pedigree.
        ibd_seg_list: List of phased segments of the form
                        [[id1, id2, hap1, hap2, chrom, start, end, cm]]
        pw_ll_cls: instance of class with methods for
                   obtaining pairwise log likelihoods.
        condition: Boolean. If true, condition on observing
                   IBD.
        max_peds: Int controlling the number of pedigrees
                  that are retained at each step.
        max_start_peds: int
            The number of pedigreees from the previous round to
            use as a basis for building pedigrees in the next round.
        min_seg_len: minimum observable segment length in cM.
        connect_up_only: when connecting two pedigrees that are
            sufficiently far apart (minimum shared IBD between
            their two closest IDs), use an
            approach similar to Bonsai v2. In other words,
            only
            look for connection points in a pedigree that are
            common ancestors of the genotyped nodes that share
            IBD with the other pedigree (id_set). If backround IBD or
            cryptic inbreeding causes id_set to have no common
            ancestor, use the ancestor with the most descendants
            in id_set. Also use the most recent such ancestor
            who has at most one parent (open ancestor).
        db_fig_base_dir: directory to save figures to for debugging.
        true_ped: true pedigree to compare to for debugging. If specified
            Bonsai will enter debug mode at certain key points when
            the true pedigree differs sufficiently from the estimated
            pedigree.

    Returns:
        idx_to_up_dct_ll_list: dict mapping indices to lists of the
                        form [[up_dct, ll], ...]
                        of up node dicts and their log likelihoods
                        sorted from most to least likely.
    """

    ibd_stats = get_ibd_stats_frozenform(ibd_seg_list)

    # remove pairwise stats for IDs in the same pedigree
    for idx, id_set in idx_to_id_set.items():
        for i1,i2 in combinations(id_set, r=2):
            key = frozenset({i1, i2})
            if key in ibd_stats:
                ibd_stats.pop(key)

    idx_to_up_dict_ll_list = copy.deepcopy(idx_to_up_dict_ll_list)
    id_to_idx = copy.deepcopy(id_to_idx)
    idx_to_id_set = copy.deepcopy(idx_to_id_set)
    step_ct = 0
    while len(idx_to_up_dict_ll_list) > 1 and ibd_stats:
        # get the closest pair of IDs that are not
        # already in the same pedigree
        c1,c2 = get_closest_pair(ibd_stats)

        # get the family indexes of the
        # two closest IDs
        idx1 = id_to_idx[c1]
        idx2 = id_to_idx[c2]

        # get up node dict lists
        up_dct_ll_list1 = idx_to_up_dict_ll_list[idx1]
        up_dct_ll_list2 = idx_to_up_dict_ll_list[idx2]

        # merge the two lists to find the most likely pairs of pedigrees
        marginal_up_dct_ll_list = merge_up_dct_ll_lists(
            up_dct_ll_list1=up_dct_ll_list1,
            up_dct_ll_list2=up_dct_ll_list2,
        )

        # get the max_peds most likely marginal likelihood
        marg_ll_set = {r[-1] for r in marginal_up_dct_ll_list}
        marg_ll_list = sorted(marg_ll_set, reverse=True)
        min_marg_ll = min(marg_ll_list[:max_start_peds])

        # print the pedigrees that were joined and the likelihoods
        # of the resulting pedigrees
        if db_fig_base_dir is not None:
            step_ct += 1
            import os
            try:
                os.mkdir(db_fig_base_dir)
            except FileExistsError:
                pass
            try:
                fig_dir = os.path.join(db_fig_base_dir, f'step_{step_ct}')
                os.mkdir(fig_dir)
            except FileExistsError:
                pass
        else:
            fig_dir = None

        # cycle over pairs of dicts and merge them
        up_dct_ll_list = []
        for up_dct1, up_dct2, marg_ll in marginal_up_dct_ll_list:

            # if pedigree is not among the top max_peds most likely
            # pedigrees measured according to likelihoods, break.
            if marg_ll < min_marg_ll:
                break

            # fill in any missing partners in the up dicts
            # so that we can explore all nodes and partners
            # for connection
            filled_up_dct1 = fill_in_partners(up_dct=up_dct1)
            filled_up_dct2 = fill_in_partners(up_dct=up_dct2)

            this_up_dct_ll_list = combine_pedigrees(
                up_dct1=filled_up_dct1,
                up_dct2=filled_up_dct2,
                pw_ll_cls=pw_ll_cls,
                ibd_seg_list=ibd_seg_list,
                condition=condition,
                max_peds=max_peds,
                max_con_pts=max_con_pts,
                min_seg_len=min_seg_len,
                restrict_connection_points=restrict_connection_points,
                connect_up_only=connect_up_only,
                fig_dir=fig_dir,
            )
            up_dct_ll_list += this_up_dct_ll_list

        # sort from most to least likely
        up_dct_ll_list = sorted(up_dct_ll_list, key=lambda x: -x[-1])

        # restrict to the top max_peds entries
        up_dct_ll_list = up_dct_ll_list[:max_peds]

        # record the pedigrees
        idx_to_up_dict_ll_list[idx1] = up_dct_ll_list
        idx_to_up_dict_ll_list.pop(idx2)

        # merge the ID sets
        idx_to_id_set[idx1] |= idx_to_id_set[idx2]
        idx_to_id_set.pop(idx2)

        # point the IDs to the proper indexes
        for i in idx_to_id_set[idx1]:
            id_to_idx[i] = idx1

        # for all pairs in the new ID set, remove them
        # from ibd_stats.
        id_set = idx_to_id_set[idx1]
        for i1,i2 in combinations(id_set, r=2):
            key = frozenset({i1,i2})
            if key in ibd_stats:
                ibd_stats.pop(key)

        if true_ped is not None:
            # cycle over all relationship pairs in new_ped
            # and go into debug mode if we encounter a pair
            # that is parent/child or full sibling in true_ped
            # and which is not parent/child or full sibling in new_ped
            # or vice versa.
            new_ped = up_dct_ll_list[0][0]
            gt_id_set = get_gt_id_set(new_ped)
            for i1,i2 in combinations(gt_id_set, r=2):
                rel = get_simple_rel_tuple(true_ped, i1, i2)
                est_rel = get_simple_rel_tuple(new_ped, i1, i2)
                stop = False
                if (
                        (get_deg(rel) <= 2 or get_deg(est_rel) <= 2) and
                        (get_deg(rel) != get_deg(est_rel))
                ):
                    stop = True
                if stop:
                    # render the pedigree and go into debug mode
                    import os

                    from .rendering import render_ped
                    new_ped = up_dct_ll_list[0][0]
                    plot_new_ped = remove_dangly_founders(up_node_dict=new_ped)
                    plot_true_ped = remove_dangly_founders(up_node_dict=true_ped)
                    debug_dir = os.path.join(fig_dir, "debug")
                    up_dct1, up_dct2, marg_ll = marginal_up_dct_ll_list[0]
                    render_ped(up_dct=up_dct1, name='up_dct1', out_dir=debug_dir)
                    render_ped(up_dct=up_dct2, name='up_dct2', out_dir=debug_dir)
                    render_ped(up_dct=plot_true_ped, name='true', out_dir=debug_dir)
                    render_ped(up_dct=plot_new_ped, name='combined_ud', out_dir=debug_dir)
                    import pdb
                    pdb.set_trace()

        # render the pedigrees
        elif fig_dir:
            import os

            from .rendering import render_ped
            new_ped = up_dct_ll_list[0][0]
            plot_new_ped = remove_dangly_founders(up_node_dict=new_ped)
            debug_dir = os.path.join(fig_dir, "debug")
            up_dct1, up_dct2, marg_ll = marginal_up_dct_ll_list[0]
            render_ped(up_dct=up_dct1, name='up_dct1', out_dir=debug_dir)
            render_ped(up_dct=up_dct2, name='up_dct2', out_dir=debug_dir)
            render_ped(up_dct=plot_new_ped, name='combined_ud', out_dir=debug_dir)
            import pdb
            pdb.set_trace()


    return idx_to_up_dict_ll_list


def combine_up_dct_ll_lists(
    idx_to_up_dct_ll_list: dict[
        int, list[
            tuple[
                dict[int, dict[int, int]],float
            ]
        ]
    ],
):
    """
    Combine all pedigrees that were not able to be combined
    into a single pedigree that may be disjoint.

    If every pedigree combined, then idx_to_up_dct_ll_list
    will have just one key and one value, where the value
    is a list of the form [(up_dct,ll),...] of up node dicts
    and their log likelihoods. However, if not all pedigrees
    combined, then idx_to_up_dct_ll_list will have multiple
    entries of this kind. These represent disjoint pedigrees
    so the inferred up node dict is really the merge of these
    dicts. Some of the ungenotyped nodes in these dicts might
    be the same, so we need to shift the nodes so that all
    ungenotyped nodes are different.

    Args:
        idx_to_up_dct_ll_list: dict mapping indices to lists of the
                        form [[up_dct, ll], ...]
                        of up node dicts and their log likelihoods
                        sorted from most to least likely.

    Returns:
        full_up_dct: a single up dict representing possibly disjoint
                pedigrees of the form
                {node : {p1 : d1, p2 : d2},...}
    """
    min_id = -1
    full_up_dct = {}
    for idx, up_dct_ll_list in idx_to_up_dct_ll_list.items():
        # get the most likely up_dct
        up_dct = up_dct_ll_list[0][0]

        # shift IDs so that negative IDs don't overlap with previous pedigrees
        up_dct, rep_dict = shift_ids(
            ped=up_dct,
            shift=min_id,
        )

        # add the up node dict to the full up node dict
        full_up_dct.update(up_dct)

        # update the minimum index
        min_id = get_min_id(full_up_dct)

    return full_up_dct


def get_sharing_ids(
    up_dct1 : dict[int, dict[int, int]],
    up_dct2 : dict[int, dict[int, int]],
    ibd_seg_list : list[tuple[int, int, int, int, str, float, float, float]],
):
    """
    Find the set of IDs in up_dct1 that share
    IBD with up_dct2 and also the set of IDs in up_dct2
    that share IBD with up_dct1.

    Args:
        up_dct1: up node dict representing pedigree 1
                 of the form
                    {node: {parent1 : deg1, parent2 : deg2}, ..}
                 Can have zero, one, or two parents per node
        up_dct2: up node dict representing pedigree 2
                 of the form
                    {node: {parent1 : deg1, parent2 : deg2}, ..}
                 Can have zero, one, or two parents per node
        ibd_seg_list: list of the form
                 [[id1, id2, hap1, hap2, chrom_str, gen_start, gen_end, seg_len_cm],...]

    Returns:
        rel_id_set1: set of IDs in pedigree 1 that share >= 1 segment with an
                 ID in pedigree 2
        rel_id_set2: set of IDs in pedigree 2 that share >= 1 segment with an
                 ID in pedigree 1
    """

    gt_id_set1 = get_gt_id_set(ped=up_dct1)
    gt_id_set2 = get_gt_id_set(ped=up_dct2)

    rel_id_set1 = set()
    rel_id_set2 = set()
    for seg in ibd_seg_list:
        i1 = seg[0]
        i2 = seg[1]

        if (i1 in gt_id_set1) and (i2 in gt_id_set2):
            rel_id_set1.add(i1)
            rel_id_set2.add(i2)
        elif (i2 in gt_id_set1) and (i1 in gt_id_set2):
            rel_id_set1.add(i2)
            rel_id_set2.add(i1)

    return rel_id_set1, rel_id_set2


def get_anc_age_pt_est(
    up_dct: dict[int, dict[int, int]],
    anc_id: int,
    pw_ll_cls: Any,
    exp_years_per_meiosis: int=30,
):
    """
    Get a simple point estimate of age a node,
    given the ages of its most proximal descendants.

    Do this by taking the number of meioses
    separating anc_id from each of its descendants.
    For each descendant, multiply the number
    of generations by the average age difference
    in a generation in years to get the difference
    in ages and then add this to the age of the
    descendant.

    Finally, take the average of these age point
    estimates for all of the most proximal genoyped
    descendants of anc_id.

    Args:
        up_dct: an up node dict of the form
            {node: {p1 : d1, p2 : d2}, ...}
            values may have 0, 1, or 2 keys
        anc_id: ID of the ancestor in question
        pw_ll_cls: pairwise likelihood class

    Returns:
        anc_age: estimated age of the ancestor
    """

    # check if anc_id itself is genotyped
    if anc_id > 0:
        return pw_ll_cls.age_dict[anc_id]

    # get the part of the pedigree that is most
    # proximal to anc_id, going up.
    prox_up_dct = trim_to_proximal(
        down_dct=up_dct,
        root=anc_id,
    )
    up_deg_dict = get_rel_deg_dict(
        node_dict=prox_up_dct,
        i=anc_id,
    )
    up_deg_dict = {r: d for r,d in up_deg_dict.items() if r > 0}  # trim to genotyped

    # get the part of the pedigree that is most
    # proximal to anc_id, going down.
    dn_dct = reverse_node_dict(up_dct)
    prox_dn_dct = trim_to_proximal(
        down_dct=dn_dct,
        root=anc_id,
    )
    dn_deg_dict = get_rel_deg_dict(
        node_dict=prox_dn_dct,
        i=anc_id,
    )
    dn_deg_dict = {r: d for r,d in dn_deg_dict.items() if r > 0}  # trim to genotyped

    # cycle over up_deg_dict and dn_deg_dict and get the
    # point estimates of ages of anc_id
    anc_id_age_est_list = []
    for r, d in up_deg_dict.items():
        age = pw_ll_cls.age_dict[r]
        age_est = age - d * exp_years_per_meiosis
        anc_id_age_est_list.append(age_est)
    for r, d in dn_deg_dict.items():
        age = pw_ll_cls.age_dict[r]
        age_est = age + d * exp_years_per_meiosis
        anc_id_age_est_list.append(age_est)

    anc_age = np.mean(anc_id_age_est_list).round().astype(int)

    return anc_age

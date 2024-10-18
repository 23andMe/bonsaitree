from typing import Optional

from .connections import (
    combine_up_dicts,
    connect_pedigrees_through_points,
    get_connecting_points_degs_and_log_likes,
)
from .constants import (
    CONNECT_UP_ONLY,
    MAX_CON_PTS,
    MAX_PEDS,
    MAX_START_PEDS,
    MEAN_BGD_LEN,
    MEAN_BGD_NUM,
    MIN_SEG_LEN,
    RESTRICT_CON_PTS,
)
from .ibd import (
    get_ibd_stats_unphased,
    get_id_set_from_seg_list,
    get_phased_to_unphased,
    get_unphased_to_phased,
)
from .likelihoods import PwLogLike


def initialize_input_dicts(
    bio_info : list[dict[str, int]],
):
    """
    For a list of individual IDs, set up the necessary data structures to
    combine them into a pedigree.

    Initialize the dictionaries that map which IDs are in which pedigrees
    and which contain the lists of smaller assembled pedigrees and their
    likelihoods.

    Returns:
        idx_to_up_dict_ll_list: Dict of the form
            {
                idx: [[up_node_dct1, ll1], ...],...
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

    """
    idx = 0
    id_to_idx = {}
    idx_to_id_set = {}
    idx_to_up_dict_ll_list = {}
    for info in bio_info:
        idx += 1
        gid = info['genotype_id']
        id_to_idx[gid] = idx
        idx_to_id_set[idx] = {gid}
        idx_to_up_dict_ll_list[idx] = [({gid : {}}, 0)]
    return idx_to_up_dict_ll_list, id_to_idx, idx_to_id_set


def build_pedigree(
    bio_info : list[dict[str, int]],
    unphased_ibd_seg_list : list[
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
    ]=None,
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
    ]=None,
    condition_pair_set : Optional[frozenset[int]] = None,
    min_seg_len : float = MIN_SEG_LEN,
    max_con_pts : int = MAX_CON_PTS,
    restrict_connection_points : bool = RESTRICT_CON_PTS,
    connect_up_only : bool = CONNECT_UP_ONLY,
    max_peds : int = MAX_PEDS,
    max_start_peds : int = MAX_START_PEDS,
    db_fig_base_dir : Optional[str] = None,
    true_ped: Optional[dict[int, dict[int, int]]]=None,
    mean_bgd_num : float=MEAN_BGD_NUM,
    mean_bgd_len : float=MEAN_BGD_LEN,
):
    """
    Build a pedigree for a list of genotyped IDs.

    Args:
        bio_info : List[Dict[str, int]]
            A list of dictionaries where each dictionary contains information about
            an individual in the pedigree. bio_info is of the form
                [{'genotype_id' : <gid>, 'sex' : <sex>, 'age' : <age>, 'coverage' : <coverage>, ...}, ...]
            where the keys shown above are the minimum required keys, although other keys
            are allowed.
        unphased_ibd_seg_list: list of the form
            [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
        phased_ibd_seg_list: list of the form
            [[id1, id2, hap1, hap2, chromosome, gen_start, gen_end, seg_cm]]
        condition_pair_set : Optional[FrozenSet[int]] If a pair is in this set, use the
            conditional likelihood. If not, use the unconditional likelihood.
        min_seg_len: float
            Minimum observable segment length.
        max_con_pts: int
            Maximum number of connection points to consider when combining two pedigrees.
            A connection point is a node on a pedigree (ungenotyped or genotyped) through
            which the pedigree can be connected to another individual or pedigree.
        restrict_connection_points: bool
            Restrict the set of connection points in a pedigree the subtree connecting
            the set of genotyped nodes that are related to the other pedigree.
        connect_up_only: bool
            Only connect two sub pedigrees upward through their common ancestors.
        max_peds: int
            Maximum number of pedigrees to consider when combining pedigrees.
        max_start_peds: int
            Maximum number of pedigrees to use as starting points for the search of MAX_PEDS pedigrees.
        db_fig_base_dir: str
            Optional directory for storing renderings of each intermediate pedigree.
        true_ped: true pedigree to compare to for debugging. If specified
            Bonsai will enter debug mode at certain key points when
            the true pedigree differs sufficiently from the estimated
            pedigree.
        mean_bgd_num: float. Mean number of background segments.
        mean_bgd_len: float. Mean length of background segments.

    Returns:
        up_dict_ll_list: List
            List of the form [[up_node_dict, log_like], ...] where up_node_dict is an
            inferred pedigree and log_like is the log likelihood of the pedigree.
            Elements of up_dict_ll_list are sorted from most to least likely.
    """  # noqa : E501

    # set up unphased IBD segments if none are provided
    if unphased_ibd_seg_list is None:
        unphased_ibd_seg_list = get_phased_to_unphased(phased_ibd_seg_list)

    # set up pseudo-phased IBD segments if none are provided
    if phased_ibd_seg_list is None:
        phased_ibd_seg_list = get_unphased_to_phased(unphased_ibd_seg_list)

    # generate PwLog_like instance for computing pairwise likelihoods
    pw_ll_cls = PwLogLike(
        bio_info=bio_info,
        unphased_ibd_seg_list=unphased_ibd_seg_list,
        condition_pair_set=condition_pair_set,
        mean_bgd_num=mean_bgd_num,
        mean_bgd_len=mean_bgd_len,
    )

    # initialize the input dictionaries
    idx_to_up_dict_ll_list, id_to_idx, idx_to_id_set = initialize_input_dicts(bio_info)

    # TODO: we'll need to compute pairwise likelihoods based the ascertainment of each pair.
    # knowledge about ascertainment can be passed in bio_info and it can be determined in
    # the function that we'll write to determine who gets placed next. For now, assume that
    # all IBD sharing is conditional on observing some.
    condition = True

    # build the pedigrees
    result = combine_up_dicts(
        idx_to_up_dict_ll_list = idx_to_up_dict_ll_list,
        id_to_idx = id_to_idx,
        idx_to_id_set = idx_to_id_set,
        ibd_seg_list = phased_ibd_seg_list,
        pw_ll_cls = pw_ll_cls,
        condition = condition,
        max_peds = max_peds,
        max_start_peds = max_start_peds,
        max_con_pts = max_con_pts,
        min_seg_len = min_seg_len,
        restrict_connection_points = restrict_connection_points,
        connect_up_only = connect_up_only,
        db_fig_base_dir = db_fig_base_dir,
        true_ped = true_ped,
    )

    # get the index of the pedigree that was built
    idx = [*result][0]

    # get the list of pedigrees and their likelihoods
    up_dict_ll_list = result[idx]

    return up_dict_ll_list


def connect_new_node_many_ways(
    node : int,
    up_node_dict : dict[int, dict[int, int]],
    bio_info : list[dict[str, int]],
    unphased_ibd_seg_list : list[
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
    ]=None,
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
    ]=None,
    condition_pair_set : Optional[frozenset[int]] = None,
    min_seg_len : float = MIN_SEG_LEN,
    max_con_pts : int = MAX_CON_PTS,
    restrict_connection_points : bool = RESTRICT_CON_PTS,
    connect_up_only : bool = CONNECT_UP_ONLY,
    max_peds : int = MAX_PEDS,
    max_start_peds : int = MAX_START_PEDS,
    true_ped: Optional[dict[int, dict[int, int]]]=None,
    mean_bgd_num : float=MEAN_BGD_NUM,
    mean_bgd_len : float=MEAN_BGD_LEN,
):
    """
    Build a pedigree for a list of genotyped IDs.

    Args:
        node: int
            The ID of the node to connect to the up_node_dict.
        up_node_dict: dict
            A dictionary of the form
                {node1 : {parent1 : 1, parent2 : 1}, ...}
            mapping each node in the pedigree to its parent IDs to the degree from the parent
            to the child. In our case, the degree is always 1.
        bio_info : List[Dict[str, int]]
            A list of dictionaries where each dictionary contains information about
            an individual in the pedigree. bio_info is of the form
                [{'genotype_id' : <gid>, 'sex' : <sex>, 'age' : <age>, 'coverage' : <coverage>, ...}, ...]
            where the keys shown above are the minimum required keys, although other keys
            are allowed.
        unphased_ibd_seg_list: list of the form
            [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
        phased_ibd_seg_list: list of the form
            [[id1, id2, hap1, hap2, chromosome, gen_start, gen_end, seg_cm]]
        condition_pair_set : Optional[FrozenSet[int]] If a pair is in this set, use the
            conditional likelihood. If not, use the unconditional likelihood.
        min_seg_len: float
            Minimum observable segment length.
        max_con_pts: int
            Maximum number of connection points to consider when combining two pedigrees.
            A connection point is a node on a pedigree (ungenotyped or genotyped) through
            which the pedigree can be connected to another individual or pedigree.
        restrict_connection_points: bool
            Restrict the set of connection points in a pedigree the subtree connecting
            the set of genotyped nodes that are related to the other pedigree.
        connect_up_only: bool
            Only connect two sub pedigrees upward through their common ancestors.
        max_peds: int
            Maximum number of pedigrees to consider when combining pedigrees.
        max_start_peds: int
            Maximum number of pedigrees to use as starting points for the search of MAX_PEDS pedigrees.
        true_ped: true pedigree to compare to for debugging. If specified
            Bonsai will enter debug mode at certain key points when
            the true pedigree differs sufficiently from the estimated
            pedigree.
        mean_bgd_num: float. Mean number of background segments.
        mean_bgd_len: float. Mean length of background segments.

    Returns:
        up_dict_ll_list: List
            List of the form [[up_node_dict, log_like], ...] where up_node_dict is an
            inferred pedigree and log_like is the log likelihood of the pedigree.
            Elements of up_dict_ll_list are sorted from most to least likely.
    """  # noqa : E501

    # set up unphased IBD segments if none are provided
    if unphased_ibd_seg_list is None:
        unphased_ibd_seg_list = get_phased_to_unphased(phased_ibd_seg_list)

    # set up pseudo-phased IBD segments if none are provided
    if phased_ibd_seg_list is None:
        phased_ibd_seg_list = get_unphased_to_phased(unphased_ibd_seg_list)

    # generate PwLog_like instance for computing pairwise likelihoods
    pw_ll_cls = PwLogLike(
        bio_info=bio_info,
        unphased_ibd_seg_list=unphased_ibd_seg_list,
        condition_pair_set=condition_pair_set,
        mean_bgd_num=mean_bgd_num,
        mean_bgd_len=mean_bgd_len,
    )

    # initialize the input dictionaries
    idx_to_up_dict_ll_list = {}
    idx_to_id_set = {}
    idx_to_up_dict_ll_list[0] = [(up_node_dict, 0)]
    idx_to_up_dict_ll_list[1] = [({node : {}}, 0)]
    idx_to_id_set[0] = {i for i in up_node_dict if i > 0}
    idx_to_id_set[1] = {node}
    id_to_idx = {i : 0 for i in idx_to_id_set[0]}
    id_to_idx[node] = 1

    # TODO: we'll need to compute pairwise likelihoods based the ascertainment of each pair.
    # knowledge about ascertainment can be passed in bio_info and it can be determined in
    # the function that we'll write to determine who gets placed next. For now, assume that
    # all IBD sharing is conditional on observing some.
    condition = True

    # build the pedigrees
    result = combine_up_dicts(
        idx_to_up_dict_ll_list = idx_to_up_dict_ll_list,
        id_to_idx = id_to_idx,
        idx_to_id_set = idx_to_id_set,
        ibd_seg_list = phased_ibd_seg_list,
        pw_ll_cls = pw_ll_cls,
        condition = condition,
        max_peds = max_peds,
        max_start_peds = max_start_peds,
        max_con_pts = max_con_pts,
        min_seg_len = min_seg_len,
        restrict_connection_points = restrict_connection_points,
        connect_up_only = connect_up_only,
        true_ped = true_ped,
    )

    # get the index of the pedigree that was built
    idx = [*result][0]

    # get the list of pedigrees and their likelihoods
    up_dict_ll_list = result[idx]

    return up_dict_ll_list


def get_new_node_connections(
    node : int,
    up_node_dict : dict[int, dict[int, int]],
    bio_info : list[dict[str, int]],
    unphased_ibd_seg_list : list[
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
    ]=None,
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
    ]=None,
    condition_pair_set : Optional[set[frozenset[int]]] = None,
    min_seg_len : float = MIN_SEG_LEN,
    max_con_pts : int = MAX_CON_PTS,
    restrict_connection_points : bool = RESTRICT_CON_PTS,
    connect_up_only : bool = CONNECT_UP_ONLY,
    condition : bool = False,
    mean_bgd_num : float=MEAN_BGD_NUM,
    mean_bgd_len : float=MEAN_BGD_LEN,
):
    """
    For a starting pedigree specified in up_node_dict and for a new relative node, find the
    set of connection points in up_node_dict, the relationships between these nodes and 'node',
    and the log likelihood of each point and relationship.

    Args:
        up_node_dct : Dict[int, Dict[int, int]]
            A dictionary where the keys are node IDs and the values are dictionaries
            mapping the IDs of the parents of the node to the degree of the parent
            to the child. In our case, the degree is always 1.
        node : int
            The ID of the node to connect to the up_node_dict.
        bio_info : List[Dict[str, int]]
            A list of dictionaries where each dictionary contains information about
            an individual in the pedigree. bio_info is of the form
                [{'genotype_id' : <gid>, 'sex' : <sex>, 'age' : <age>, 'coverage' : <coverage>, ...}, ...]
            where the keys shown above are the minimum required keys, although other keys
            are allowed.
        unphased_ibd_seg_list: list of the form
            [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
        phased_ibd_seg_list: list of the form
            [[id1, id2, hap1, hap2, chromosome, gen_start, gen_end, seg_cm]]
        condition_pair_set : Optional[Set[FrozenSet[int]]]
            Has the form {frozenset({i1,i2}),...}. If a pair is in this set, use the
            conditional likelihood. If not, use the unconditional likelihood.
        min_seg_len: float
            Minimum observable segment length.
        max_con_pts: int
            Maximum number of connection points to consider when combining two pedigrees.
            A connection point is a node on a pedigree (ungenotyped or genotyped) through
            which the pedigree can be connected to another individual or pedigree.
        restrict_connection_points: bool
            Restrict the set of connection points in a pedigree the subtree connecting
            the set of genotyped nodes that are related to the other pedigree.
        connect_up_only: bool
            Only connect two sub pedigrees upward through their common ancestors.
        condition: bool
            Condition on the event that IBD was observed.
        mean_bgd_num: float. Mean number of background segments.
        mean_bgd_len: float. Mean length of background segments.

    Returns:
        anc_deg_log_like_list: List of the form
                [[con_pt, rel_tuple, log_like],...]
            ordered from most to least likely.

                con_pt: The point in up_node_dict through which up_node_dict is connected to node.

                    con_pt is a tuple of the form (p,partner,dir), where p is the primary node
                    in up_node_dict that is connected to node. partner is a potential partner of p.
                    partner is set to None if a no partner of p is connected to node.

                    dir is the direction of the connection. dir is set to 1 if p connects to
                    node through a lineage extending upward from p. dir is set to 0 if p
                    connects to node through a lineage extending downward from p. dir is set
                    to None if p connects "on" node (i.e., if p and node are the same node).

                rel_tuple: is a relationship tuple of the form (up, down, num_ancs) where
                    up is the degree from p up to its common ancestor(s) with node, down is the
                    degree from the common ancestor(s) down node, and num_ancs is the number
                    of common ancestors between p and node.

                log_like: log likelihood of the connection.
    """  # noqa : E501

    # set up unphased IBD segments if none are provided
    if unphased_ibd_seg_list is None:
        unphased_ibd_seg_list = get_phased_to_unphased(phased_ibd_seg_list)

    # set up pseudo-phased IBD segments if none are provided
    if phased_ibd_seg_list is None:
        phased_ibd_seg_list = get_unphased_to_phased(unphased_ibd_seg_list)

    # generate PwLog_like instance for computing pairwise likelihoods
    pw_ll_cls = PwLogLike(
        bio_info=bio_info,
        unphased_ibd_seg_list=unphased_ibd_seg_list,
        condition_pair_set=condition_pair_set,
        mean_bgd_num=mean_bgd_num,
        mean_bgd_len=mean_bgd_len,
    )

    # set up up_node_dicts to combine
    up_dct1 = up_node_dict
    up_dct2 = {node : {}}

    # TODO: we'll need to compute pairwise likelihoods based the ascertainment of each pair.
    # knowledge about ascertainment can be passed in bio_info and it can be determined in
    # the function that we'll write to determine who gets placed next. For now, assume that
    # all IBD sharing is conditional on observing some.
    #condition = False

    anc_deg_log_like_list = get_connecting_points_degs_and_log_likes(
        up_dct1=up_dct1,
        up_dct2=up_dct2,
        pw_ll_cls=pw_ll_cls,
        ibd_seg_list=phased_ibd_seg_list,
        condition=condition,
        min_seg_len=min_seg_len,
        max_con_pts=max_con_pts,
        restrict_connection_points=restrict_connection_points,
        connect_up_only=connect_up_only,
    )

    # remove point2 to avoid confusion for downstream users
    anc_deg_log_like_list = [[p1,rel,log_like] for p1,p2,rel,log_like in anc_deg_log_like_list]

    return anc_deg_log_like_list


def connect_new_node(
    node: int,
    up_node_dict: dict[int, dict[int, int]],
    con_pt: tuple[int, Optional[int], Optional[int]],
    rel_tuple: tuple[int, int, int],
):
    """
    Connect a new node to a pedigree through a connection point.

    Args:
        node: the ID of the new node to connect to the pedigree.
        up_node_dict: dict
            A dictionary of the form
                {node1 : {parent1 : 1, parent2 : 1}, ...}
        con_pt: a point of the form (p,partner,dir) where p is the primary node
                in up_node_dict that is connected to node. partner is a potential
                partner of p. Partner is set to None if a no partner of p is
                connected to node.

                dir (unused) is the direction of the connection. dir is set to 1 if p
                connects to node through a lineage extending upward from p. dir is set
                to 0 if p connects to node through a lineage extending downward from p.
                dir is set to None if p connects "on" node (i.e., if p and node are the
                same node).

                con_pt is the first entry in the tuple (point, rel_tuple, log_like)
                returned by get_new_node_connections.
        rel_tuple: a relationship tuple of the form (up, down, num_ancs).
    """
    # get IDs we are connecting
    id1, pid1, _ = con_pt
    id2 = node
    pid2 = None

    # get the relationship of the connection
    deg1, deg2, num_ancs = rel_tuple

    # get up dicts to connect
    up_dct1 = up_node_dict
    up_dct2 = {node: {}}

    # connect up dicts
    new_ped_list = connect_pedigrees_through_points(
        id1=id1,
        id2=id2,
        pid1=pid1,
        pid2=pid2,
        up_dct1=up_dct1,
        up_dct2=up_dct2,
        deg1=deg1,
        deg2=deg2,
        num_ancs=num_ancs,
        simple=True,
    )

    # return the pedigree
    if len(new_ped_list) == 0:
        return {}
    else:
        return new_ped_list[0]


def get_next_node(
    placed_id_set: set[int],
    unphased_ibd_seg_list : list[
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
    ]=None,
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
    ]=None,
):
    """
    Get the next node to add to a pedigree.

    Args:
        placed_id_set: set of genotype IDs that are already placed in the pedigree.
        phased_ibd_seg_list: list of the form
            [[id1, id2, hap1, hap2, chromosome, gen_start, gen_end, seg_cm]]
        unphased_ibd_seg_list: list of the form
            [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]

    Returns:
        (node, pid, max_ibd): Tuple[int, int, int]
            node: the ID of the next node to add to the pedigree.
            pid: the ID of the node in the pedigree that shares the most IBD with node.
            max_ibd: the amount of IBD shared between node and pid.
    """
    # set up unphased IBD segments if none are provided
    if unphased_ibd_seg_list is None:
        unphased_ibd_seg_list = get_phased_to_unphased(phased_ibd_seg_list)

    # generate PwLog_like instance for computing pairwise likelihoods
    ibd_stats = get_ibd_stats_unphased(unphased_ibd_seg_list)

    # get all IDs, placed or unplaced
    all_id_set = get_id_set_from_seg_list(unphased_ibd_seg_list)

    # get unplaced ID set
    unplaced_id_set = all_id_set - placed_id_set

    # find the unplaced ID that shares the most IBD with any placed ID
    id_max_ibd = []
    for uid in unplaced_id_set:
        max_ibd = 0
        for pid in placed_id_set:
            key = frozenset({uid, pid})
            if key in ibd_stats:
                total_half = ibd_stats[key]['total_half']
                total_full = ibd_stats[key]['total_full']
                total_ibd = total_half + total_full
                max_ibd = max(max_ibd, total_ibd)
        id_max_ibd.append((uid, pid, max_ibd))
    id_max_ibd = sorted(id_max_ibd, key=lambda x: x[-1], reverse=True)

    if len(id_max_ibd) == 0:
        return None, None, None
    else:
        return id_max_ibd[0]

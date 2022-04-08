from typing import Any, Dict, List, Set, Tuple, FrozenSet, Optional, NoReturn

from collections import defaultdict
import numbers
import warnings

import funcy as fn
import numpy as np
from itertools import combinations
import copy

from .exceptions import InconsistentSexException
from .pedigree_object import PedigreeObject

from .utils import transform_segment_lists_to_ibd_summaries
from .point_predictor import construct_point_prediction_group, point_predictions
from .distributions import load_distributions
from .build_pedigree import infer_local_pedigrees, assemble_local_pedigrees
from .relationship_preprocessing_tools import find_twins, remove_twins
from .connect_pedigree_tools import check_pred_deg_likes

INF = float('inf')

INVERT_SEX = {
    'F': 'M',
    'M': 'F',
    None: None,
}

MAX_RADIUS = INF
MAX_ADD_DEGREE = 3
MIN_REL_APPEND_TYPES = 1
MAX_REL_APPEND_TYPES = 3
IBD_THRESHOLD = 10
PED_SAVE_LIKE_DELTA_FRACTION = 0.001
PED_SAVE_LIKE_ABS_FRACTION = 0.01
NUM_SMALL_PED_OBJS_TO_SAVE = 10
REMOVE_DISTANT_THRESHOLD = INF
DROP_IBD_ALPHA = 1e-4


def build_pedigree(
    ibd_seg_list : List[List[Any]],
    bio_info : List[Dict[str,Any]],
    focal_id : Optional[int] = None,
    seed_pedigree_list : Tuple[Any, ...] = (),
    validated_node_set_list : Tuple[Set[int], ...] = (),
    ignore_validated : bool = True,
    childless_partners : Any = None,
    ibd_stats : Dict[FrozenSet[int],Dict[str,Any]] = None,
    disallow_distant_half_rels : bool = False,
    throw_unlikely_ped_exception : bool = False,
    max_radius : int = MAX_RADIUS,
    max_add_degree : int = MAX_ADD_DEGREE,
    min_rel_append_types : int = MIN_REL_APPEND_TYPES,
    max_rel_append_types : int = MAX_REL_APPEND_TYPES,
    ibd_threshold : int = IBD_THRESHOLD,
    remove_distant_threshold = REMOVE_DISTANT_THRESHOLD,
    ped_save_like_delta_fraction : float = PED_SAVE_LIKE_DELTA_FRACTION,
    ped_save_like_abs_fraction : float = PED_SAVE_LIKE_ABS_FRACTION,
    num_small_ped_objs_to_save : int = NUM_SMALL_PED_OBJS_TO_SAVE,
    drop_ibd_alpha : float = DROP_IBD_ALPHA,
):
    """
    Wrapper function for running the pedigree builder.
    Args:
        focal_id : (int) id of the focal individual for whom to build the tree
        ibd_seg_list : list of the form [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
        bio_info : List of the form [{'genotype_id : id', 'age' : age, 'sex' : sex}]
        seed_pedigree_list : List of prior pedigrees the form [normed_pedigree1, pedigree2, ...]
        validated_node_set_list : List of the form [set1, set2, ...] of genotyped individuals in seed_pedigree_list
        childless_partners : Dict mapping individuals to their unrelated partners if they have no children
        ibd_stats : Dict mapping pairs frozenset({id1,id2}) to summmary stats of IBD shared between them
        disallow_distant_half_rels : (bool) disallow half half relatives avuncular and higher unless parents exist to resolve them
        ignore_validated : (bool) ignore input "prior" pedigrees
        max_radius : (int) maximum degree of any genotyped id to the focal ID in a small pedigree
        max_add_degree : (int) largest degree we try between an unplaced ID and their closest genotyped relative
        min_rel_append_types : (int) minimum number of ways we try to place an individual relative to closest relative
        max_rel_append_types : (int) maximum number of ways we try to place an individual relative to closest relative
        ibd_threshold : (int) filter out any segment smaller than this (in cM)
        ped_save_like_delta_fraction : (float) if L(p^(k)) >= L(p^(k-1)) * ped_save_like_delta_fraction, keep it
        ped_save_like_abs_fraction : (float) if L(p^(k)) >= L(p^(1)) * ped_save_like_abs_fraction, keep it
        num_small_ped_objs_to_save : (int) number of smalll pedigree objects to save
        drop_ibd_alpha : p-value threshold for hypothesis test of dropping background IBD
    """

    # validate input
    validate_input(focal_id, ibd_seg_list, bio_info)   

    if ibd_threshold:
        ibd_seg_list = [s for s in ibd_seg_list if s[-1] >= ibd_threshold]

    if focal_id is None:
        focal_id = get_focal_id(ibd_seg_list)

    profile_information = {info['genotype_id'] : {'age' : info['age'], 'sex' : info['sex']} for info in bio_info}
    sex_dict = {info['genotype_id'] : info['sex'] for info in bio_info}
    age_dict = {info['genotype_id'] : info['age'] for info in bio_info}
    ibd_stat_dict = ibd_stats or transform_segment_lists_to_ibd_summaries(ibd_seg_list)

    # if ignoring seed pedigree information, initialize variables to empty
    if ignore_validated:
        seed_pedigree_list = None
        validated_node_set_list = None
        twin_set_dict = {}
        twin_id_to_set_dict = {}
        placed_twin_to_del_twin_set_dict = {}
        del_twin_to_placed_twin_dict = {}
        pw_rels = {}
        pw_log_likes = {}

    # if building a tree for only one individual, initialize to empty
    if len(sex_dict) == 1:
        sex = sex_dict[focal_id]
        age = age_dict[focal_id]
        up_dict = {focal_id : [sex,age]}
        ped_obj = PedigreeObject(
            up_pedigree_dict=up_dict,
            ibd_stats=ibd_stat_dict,
            age_dict=age_dict,
            sex_dict=sex_dict,
        )
        ped_obj_list = [ped_obj]
        est_index_to_ped_obj_list = {0 : [ped_obj]}
        est_index_to_gtid_set = {0 : {focal_id}}
        original_index_to_gtid_set = {0 : {focal_id}}
        original_index_to_ped_obj_list = {0 : [ped_obj]}
        combine_trace = []
    else:
        distributions = load_distributions()
        point_pred_group = construct_point_prediction_group(profile_information, ibd_stat_dict)
        pw_rels,pw_log_likes = point_predictions(
            point_prediction_group = point_pred_group,
            distribution_models = distributions,
        )

        twin_set_dict, twin_id_to_set_dict = find_twins(
            pw_rels = pw_rels,
            ibd_stat_dict = ibd_stat_dict,
            sex_dict = sex_dict,
            age_dict = age_dict,
        )

        placed_twin_to_del_twin_set_dict, del_twin_to_placed_twin_dict, pw_rels, pw_log_likes = remove_twins(
            twin_set_dict = twin_set_dict,
            pw_rels = pw_rels,
            pw_log_likes = pw_log_likes,
            keep_set = {focal_id},
        )

        # subset sex_dict, age_dict, and ibd_stat_dict down to the set with all but one member of each twin set removed.
        keep_id_set = {*pw_rels}
        sex_dict = {iid : sex for iid,sex in sex_dict.items() if iid in keep_id_set}
        age_dict = {iid : age for iid,age in age_dict.items() if iid in keep_id_set}
        ibd_stat_dict = filter_self_ibd(ibd_stat_dict)

        result = infer_local_pedigrees(
            focal_id = focal_id,
            sex_dict = sex_dict,
            age_dict = age_dict,
            pw_rels = pw_rels,
            pw_log_likes = pw_log_likes,
            ibd_stat_dict = ibd_stat_dict,
            max_radius = INF,
            max_add_degree = max_add_degree,
            min_rel_append_types = min_rel_append_types,
            max_rel_append_types = max_rel_append_types,
            ped_save_log_like_delta_fraction = ped_save_like_delta_fraction,
            ped_save_log_like_abs_fraction = ped_save_like_abs_fraction,
            disallow_distant_half_rels = disallow_distant_half_rels,
            use_age_info = True,
            seed_pedigree_list = seed_pedigree_list,
            validated_node_set_list = validated_node_set_list,
        )
        index_to_gtid_set, index_to_ped_obj_list, gtid_to_ped_obj_index, traces_dict = result

        # store a copy of the original component trees we are combining
        original_index_to_gtid_set = copy.deepcopy(index_to_gtid_set)
        original_index_to_ped_obj_list = copy.deepcopy(index_to_ped_obj_list)

        # assemble pedigree using the new big pedigree builder.
        ped_obj_list,est_index_to_ped_obj_list,est_index_to_gtid_set,combine_trace = assemble_local_pedigrees(
            focal_id = focal_id,
            ibd_seg_list = ibd_seg_list,
            ibd_stat_dict = ibd_stat_dict,
            pw_rels = pw_rels,
            pw_log_likes = pw_log_likes,
            index_to_gtid_set = index_to_gtid_set,
            index_to_ped_obj_list = index_to_ped_obj_list,
            gtid_to_ped_obj_index = gtid_to_ped_obj_index,
            num_ped_objs = num_small_ped_objs_to_save,
            disallow_distant_half_rels = disallow_distant_half_rels,
            drop_ibd_alpha = drop_ibd_alpha,
        )

        ped_obj = ped_obj_list[0]
        placed_gt_id_list = [uid for uid in ped_obj.up_pedigree_dict.keys() if not (isinstance(uid,int) and uid < 0)]
        consistent_pw_likes,inconsistent_pair_info = check_pred_deg_likes(
            focal_id = focal_id,
            pw_rels = pw_rels,
            pw_log_likes = pw_log_likes,
            ped_pw_rels = ped_obj.rels,
            placed_gt_id_list = placed_gt_id_list,
            age_dict = age_dict,
            radius_deg=3,
            deg_diff=1,
            user_error_detected=False,
            throw_exception=throw_unlikely_ped_exception,
        )

        inconsistent_sexes,inconsistent_sex_node = ped_obj.inconsistent_sexes()
        if inconsistent_sexes:
            raise InconsistentSexException("Inconsistent sexes near {}.".format(inconsistent_sex_node))

    ped_obj.fill_in_parents()
    
    # map ungenotyped nodes from seed pedigrees and back onto output pedigrees
    if seed_pedigree_list:
        ped_obj, unplaced_validated_node_set, placed_validated_node_set, matched_id_dict_new_to_old, matched_id_dict_old_to_new \
        = replace_validated_nodes(
            focal_id,
            ped_obj,
            seed_pedigree_list,
            validated_node_set_list,
            age_dict,
            sex_dict
        )
    else:
        unplaced_validated_node_set = set()
        placed_validated_node_set = set()
        matched_id_dict_old_to_new = dict()
        matched_id_dict_new_to_old = dict()

    ped_obj = remove_distant(    
        po = ped_obj,
        focal_id = focal_id,
        matched_id_dict_old_to_new = matched_id_dict_old_to_new,
        placed_validated_node_set = placed_validated_node_set,
        remove_distant_threshold = remove_distant_threshold,
    )

    normed_pedigree = normalize_pedigree(
        focal_id=focal_id,
        ped_obj=ped_obj,
        up_pedigree_dict=ped_obj.up_pedigree_dict,
        sex_dict=sex_dict,
        del_twin_to_placed_twin=del_twin_to_placed_twin_dict,
    )

    return_data = dict()

    # return info about the pedigree topology
    return_data['normed_pedigree'] = normed_pedigree
    return_data['ped_obj'] = ped_obj
    return_data['ped_obj_list'] = ped_obj_list
    return_data['est_index_to_ped_obj_list'] = est_index_to_ped_obj_list
    return_data['est_index_to_gtid_set'] = est_index_to_gtid_set

    # return information about parental side
    normed_up_dict = build_up_pedigree_dict_from(normed_pedigree, age_dict)
    normed_ped_obj = PedigreeObject(normed_up_dict)
    return_data['parental_side_dict'] = get_parental_side_dict(focal_id, normed_ped_obj, del_twin_to_placed_twin_dict)    

    # return info about twins and partners
    return_data['twin_dict'] = placed_twin_to_del_twin_set_dict
    return_data['childless_partners'] = childless_partners

    # return info about placed relatives in seed pedigrees
    return_data['unplaced_validated_node_set'] = unplaced_validated_node_set
    return_data['matched_id_dict_old_to_new'] = matched_id_dict_old_to_new
    return_data['matched_id_dict_new_to_old'] = matched_id_dict_new_to_old

    # return info about hiccups in the build process
    return_data['unplaced_validated_node_set'] = unplaced_validated_node_set
    return_data['rerun_from_scratch'] = False
    return_data['user_error_detected'] = False

    # return info that's helpful for debugging
    return_data['original_index_to_gtid_set'] = original_index_to_gtid_set
    return_data['original_index_to_ped_obj_list'] = original_index_to_ped_obj_list

    # return the focal ID so pedigrees can be compared with different focal IDs
    return_data['focal_id'] = focal_id

    return return_data


def get_focal_id(ibd_seg_list):
    share_dict = defaultdict(int)
    for seg in ibd_seg_list:
        i1,i2 = seg[:2]
        share_dict[i1] += seg[-1]
        share_dict[i2] += seg[-1]
    return max(share_dict.items(), key=lambda x: x[1])[0]


def remove_distant(
    po : Any,
    focal_id : int,
    matched_id_dict_old_to_new : Dict[int, int],
    placed_validated_node_set : Set[int],
    remove_distant_threshold : int,
) -> Any:
    """
    Remove all distant relatives if the degree (up, down, num_ancs) from the
    focal id to the relatives satisfies up > remove_distant_threshold.

    Args:
        po : instance of PedigreeObject()
        focal_id : int
        matched_id_dict_old_to_new : {id2 : id1} where id2 in po2 is matched to id1 in po1
        placed_validated_node_set : set of placed validated genotyped ids.
        remove_distant_threshold : int
    """    
    po_copy = copy.deepcopy(po) # avoid modifying inputs
    if remove_distant_threshold < INF:
        placed_genotyped_id_set = {i for i in po_copy.up_pedigree_dict if i > 0} | {i for i in po_copy.down_pedigree_dict if i > 0}
        keep_set = {i for i in placed_genotyped_id_set if po_copy.rels[focal_id][i][0] <= remove_distant_threshold}
        keep_set |= {focal_id}
        keep_set |= {matched_id_dict_old_to_new[uid] for uid in placed_validated_node_set} # new names of the placed validated nodes
        po_copy.keep_nodes(keep_gt_node_set=keep_set)
    return po_copy


def filter_self_ibd(
    ibd_stat_dict : Dict[FrozenSet[int],Dict[str,Any]] = None,
) -> Dict[FrozenSet[int],Dict[str,Any]]:
    """
    Remove entries from ibd_stat_dict that show self shared IBD
    """
    new_ibd_stat_dict = defaultdict(lambda: {
        'total_half': 0,
        'total_full': 0,
        'num_half': 0,
        'max_seg_cm': 0}
    )
    for key,val in ibd_stat_dict.items():
        if len(key) == 2:
            new_ibd_stat_dict[key] = val
    return new_ibd_stat_dict


def build_up_pedigree_dict_from(
    pedigree : Dict[int, Tuple[str, int, int]],
    age_dict : Dict[int, int],
) -> Dict[int, List[Any]]:
    """
    # Args
    pedigree:
        {genotype_id: (sex, parent_1, parent_2)}
    age_dict:
        {genotype_id : age}

    # Returns
        {genotype_id: [sex, age, p1, p2]}
    """
    return {
        uid: [sex, age_dict.get(uid)] + list(filter(None, [parent_1, parent_2]))
        for uid, (sex, parent_1, parent_2)
        in pedigree.items()
    }


def normalize_pedigree(
    focal_id : int,
    ped_obj : Any,
    up_pedigree_dict : Dict[int, List[Any]],
    sex_dict : Dict[int, str],
    del_twin_to_placed_twin : Dict[int, int],
) -> Dict[int, Tuple[str, int, int]]:
    """
    Extracts a normalized pedigree and imputes sexes when possible
    - Fills in missing parents with None
    - Adds twins back
    - Imputes spouse sex
    - Ensures out to great grandparent nodes exist for focal id
    - Deletes unrelated leaf nodes

    # Returns
        {genotype_id: (sex, parent_1, parent_2)}
    """

    ped_obj.fill_in_parents()
    ped_obj.order_sexes()

    predicted_relationships = []
    # Normalize results
    for uid, predicted_rels in up_pedigree_dict.items():
        missing_parents = [None] * (4 - len(predicted_rels))
        sex, age, parent_1, parent_2 = predicted_rels + missing_parents
        predicted_relationships.append([uid, sex, age, parent_1, parent_2])

    # Build result dictionary
    predicted_pedigree = {}
    ped_obj.update_all_ids()
    extended_sex_dict = {uid : sex_dict.get(uid) for uid in ped_obj.all_ids} # sex dict including ungenotyped people
    inferred_sexes = infer_sexes(extended_sex_dict, predicted_relationships)
    for uid, _, age, parent_1, parent_2 in predicted_relationships:
        predicted_pedigree[uid] = (inferred_sexes[uid], parent_1, parent_2)

    # Fill in great grandparents
    if focal_id not in del_twin_to_placed_twin:
        predicted_pedigree = fill_ancestors(focal_id, predicted_pedigree, generations=3)
    else:
        twin_id = del_twin_to_placed_twin[focal_id]
        predicted_pedigree = fill_ancestors(twin_id, predicted_pedigree, generations=3)

    # Add twins back
    for unplaced_twin_uid, placed_twin_uid in del_twin_to_placed_twin.items():
        if placed_twin_uid in predicted_pedigree:
            sex, parent_1, parent_2 = predicted_pedigree[placed_twin_uid]
            predicted_pedigree[unplaced_twin_uid] = (sex, parent_1, parent_2)

    # Delete unreachable nodes (can happen when we resumed a pedigree)
    if len(up_pedigree_dict) > 1:
        referenced_ids = set()
        for uid, (_, parent_1, parent_2) in predicted_pedigree.items():
            if parent_1 or parent_2:
                referenced_ids.add(uid)
            referenced_ids.add(parent_1)
            referenced_ids.add(parent_2)
        orphaned_ids = set(predicted_pedigree.keys()) - referenced_ids
        for orphaned_id in orphaned_ids:
            predicted_pedigree.pop(orphaned_id)

    return predicted_pedigree


def get_parental_side_dict(
    focal_id : int,
    ped_obj : Any, 
    del_twin_to_placed_twin_dict : Dict[int, int],
) -> Dict[int, str]:
    """
    Get a dictionary mapping each id in the pedigree to the parental side: 'mother', 'father', or 'both'
    Args:
        focal_id : (int)
        ped_obj : instance of PedigreeObject class
        del_twin_to_placed_twin_dict : Dict mapping unplaced twins to their placed counterparts
    """

    pid1,pid2 = ped_obj.up_pedigree_dict[focal_id][2:]
    pid1_sex = ped_obj.up_pedigree_dict.get(pid1,[None])[0]
    pid2_sex = ped_obj.up_pedigree_dict.get(pid2,[None])[0]

    rel_ids1 = ped_obj.rel_dict[pid1]['anc'] | ped_obj.rel_dict[pid1]['rel'] | (ped_obj.rel_dict[pid1]['desc'] - ped_obj.rel_dict[pid2]['desc'])       
    rel_ids2 = ped_obj.rel_dict[pid2]['anc'] | ped_obj.rel_dict[pid1]['rel'] | (ped_obj.rel_dict[pid2]['desc'] - ped_obj.rel_dict[pid1]['desc'])
    rel_ids1.add(pid1)
    rel_ids2.add(pid2)
    both_ids = ped_obj.rel_dict[pid1]['desc'] & ped_obj.rel_dict[pid2]['desc']

    rel_ids_list1 = [*rel_ids1]
    for iid in rel_ids_list1:
        rel_ids1 |= ped_obj.rel_dict[iid]['anc']

    rel_ids_list2 = [*rel_ids2]
    for iid in rel_ids_list2:
        rel_ids2 |= ped_obj.rel_dict[iid]['anc']

    parental_side_dict = {uid : None for uid in ped_obj.all_ids}
    if pid1_sex == 'M' or pid2_sex == 'F':
        parental_side_dict.update({i : 'father' for i in rel_ids1})
        parental_side_dict.update({i : 'mother' for i in rel_ids2})
    elif pid1_sex == 'F' or pid2_sex == 'M':
        parental_side_dict.update({i : 'mother' for i in rel_ids1})
        parental_side_dict.update({i : 'father' for i in rel_ids2})
    parental_side_dict.update({i : 'both' for i in both_ids})

    for del_twin,placed_twin in del_twin_to_placed_twin_dict.items():
        if placed_twin in parental_side_dict:
            parental_side_dict[del_twin] = parental_side_dict[placed_twin]

    return parental_side_dict


def get_next_predicted_node_id(
    pedigree : Dict[int, Tuple[str, int, int]],
) -> int:
    """
    Return the minimum ungenotyped node, minus 1.
    Args:
        pedigree : {genotype_id : (sex, parent_id1, parent_id2)}
    """
    fake_nodes = [i for i in pedigree.keys() if i < 0]

    if fake_nodes:
        return min(fake_nodes) - 1
    return -1


def fill_ancestors(
    focal_id : int,
    pedigree : Dict[int, Tuple[str, int, int]],
    generations : int,
) ->  Dict[int, Tuple[str, int, int]]:
    """
    Fill in both parents for each ancestral individual
    to create a pedigree extending `generations` generations
    back from focal_id.
    Args:
        focal_id : int
        pedigree : {genotype_id : (sex, parent_id1, parent_id2)}
        generations : (int)
    """

    if not generations:
        return pedigree
    sex, parent_1, parent_2 = pedigree[focal_id]
    if not parent_1:
        parent_1 = get_next_predicted_node_id(pedigree)
        pedigree[parent_1] = (None, None, None)
    if not parent_2:
        parent_2 = get_next_predicted_node_id(pedigree)
        pedigree[parent_2] = (None, None, None)
    pedigree[focal_id] = (sex, parent_1, parent_2)
    fill_ancestors(parent_1, pedigree, generations - 1)
    return fill_ancestors(parent_2, pedigree, generations - 1)


def infer_sexes(
    sex_dict : Dict[int, str],
    predicted_relationships : Dict[int, Tuple[str, int, int, int]],
) -> Dict[int, str]:
    """
    Infer the sexes of individuals whose partners have known sexes.
    Return a dict mapping id to sex.
    Args:
        sex_dict : {genotype_id : sex}
        predicted_relationships : {genotype_id : (sex, age, parent_id1, parent_id2)}
    """

    inferred_sexes = sex_dict.copy()
    spouses = defaultdict(set)
    for uid, _, age, parent_1, parent_2 in predicted_relationships:
        spouses[parent_1].add(parent_2) # only works because we filled in parents earlier. otherwise parent id could be None
        spouses[parent_2].add(parent_1)
        # Default unknown sexes to None
        if uid not in inferred_sexes:
            inferred_sexes[uid] = None

    def infer_spouses_sex(sex, spouses_to_infer):
        for spouse in spouses_to_infer:
            if not inferred_sexes.get(spouse):
                inferred_sex = INVERT_SEX[sex]
                inferred_sexes[spouse] = inferred_sex
                infer_spouses_sex(inferred_sex, spouses[spouse])

    inferred_sexes_copy = copy.deepcopy(inferred_sexes)
    for uid, sex in inferred_sexes_copy.items():
        if sex:
            infer_spouses_sex(sex, spouses[uid])

    return inferred_sexes


def flag_ancestors(
    uid : int,
    po : Any,
    mismatch_set : Set[int],
) -> NoReturn:
    """
    WARNING: Modifies mismatch_set

    Flag ancestors who cannot be mapped between two pedigrees.

    Add uid and all ungenotyped ancestors of uid to mismatch_set, up to the next genotyped ancestor.

    Args:
        uid: Int
        po: PedigreeObject
        mismatch_set: Set
    """
    mismatch_set.add(uid)
    pids = po.up_pedigree_dict[uid][2:]
    for pid in pids:
        if not (isinstance(pid,int) and pid < 0):
            flag_ancestors(pid,po,mismatch_set)


def check_relative_concordance(
    uid1 : int,
    uid2 : int,
    po1 : Any,
    po2 : Any,
    gt_set1 : Set[int],
    gt_set2 : Set[int],
    rel_set1 : Set[int],
    rel_set2 : Set[int],
    strictness : int = 0,
) -> bool:
    """
    Check that genotyped relatives and ancestors of uid1 in po1 are also genotyped relatives and ancestors in tree 2.
    Make sure no rels/ancs in tree one are not rels/ancs in tree 2 and vice versa.

    Args:
        uid1 : Int : putative match of uid2 in po1
        uid2 : Int : putative match of uid1 in po2
        po1 : PedigreeObject : first tree
        po2 : PedigreeObject : second tree
        gt_set1 : Set : set of genotyped IDs in po1
        gt_set2 : Set : set of genotyped IDs in po2
        rel_set1 : Set : set of relatives/descendants of uid1 in po1
        rel_set2 : Set : set of relatives/descendants of uid2 in po2
        strictness : Int : values 0,1,2. Sets how strict we want the matching to be between the two trees
    """
    concordance = True
    diff_set1 = rel_set1 - rel_set2
    diff_set2 = rel_set2 - rel_set1
    if (diff_set1 & gt_set2) or (diff_set2 & gt_set1): # Then relatives have been rearranged between peds
        concordance = False

    overlap_set = rel_set1 & rel_set2

    if strictness > 0:
        for rel_id in overlap_set:
            deg1 = po1.rels[uid1][rel_id]
            deg2 = po2.rels[uid2][rel_id]
            if deg1 != deg2:
                concordance = False
                break

    if strictness > 1:
        for id1,id2 in combinations(overlap_set, r=2):
            deg1 = po1.rels[id1][id2]
            deg2 = po2.rels[id1][id2]
            if deg1 != deg2:
                concordance = False
                break

    return concordance


def update_match_dicts(
    id1 : int,
    id2 : int,
    po1 : Any,
    po2 : Any,
    gt_set1 : Set[int],
    gt_set2 : Set[int],
    matched_id_dict_1_2 : Dict[int, int],
    matched_id_dict_2_1 : Dict[int, int],
    mismatch_set1 : Set[int],
    mismatch_set2 : Set[int],
) -> bool:
    """
    WARNING: updates the following:
        mismatch_set1
        mismatch_set2
        matched_id_dict_1_2
        matched_id_dict_2_`

    Update match dicts to macth individuals across pedigrees,
    but if a discordance is found (an ID was already matched
    to some other ID, flag the individual and all their ancestors
    and return False)

    Args:
        id1 : (int) id in pedigree1 we are matching to id2 in pedigree 2
        id2 : (int) id in pedigree2 we are matching to id1 in pedigree 1
        po1 : PedigreeObject for pedigree1
        po2 : PedigreeObject for pedigree2
        gt_set1 : all genotyped ids in pedigree 1
        gt_set2 : all genotyped ids in pedigree 2
        matched_id_dict_1_2 : dict of the form {id1 : id2} mapping ids in pedigree 1 to their matches in pedigree 2
        matched_id_dict_2_1 : dict of the form {id2 : id1} mapping ids in pedigree 2 to their matches in pedigree 1
        mismatch_set1 : set of unmatchable ids in pedigree 1
        mismatch_set2 : set of unmatchable ids in pedigree 2
    """
    succeeded = True
    if id1 in matched_id_dict_1_2 and matched_id_dict_1_2[id1] != id2:
        flag_ancestors(id1,po1,mismatch_set1)
        flag_ancestors(id2,po2,mismatch_set2)
        succeeded = False
    elif id2 in matched_id_dict_2_1 and matched_id_dict_2_1[id2] != id1:
        flag_ancestors(id1,po1,mismatch_set1)
        flag_ancestors(id2,po2,mismatch_set2)
        succeeded = False
    else:
        matched_id_dict_1_2[id1] = id2
        matched_id_dict_2_1[id2] = id1

    return succeeded


def match_ancestors(
    uid1 : int,
    uid2 : int,
    po1 : Any,
    po2 : Any,
    gt_set1 : Set[int],
    gt_set2 : Set[int],
    matched_id_dict_1_2 : Dict[int, int],
    matched_id_dict_2_1 : Dict[int, int],
    mismatch_set1 : Set[int],
    mismatch_set2 : Set[int],
    strictness : int = 2,
) -> NoReturn:
    """
    WARNING: updates the following:
        mismatch_set1
        mismatch_set2
        matched_id_dict_1_2
        matched_id_dict_2_`

    Match uid1 in pedigree 1 and uid2 in pedigree 2 and then
    recursively match the ancestors of uid1 and uid2, breaking
    when we can no longer match people

    Args:
        id1 : (int) id in pedigree1 we are matching to id2 in pedigree 2
        id2 : (int) id in pedigree2 we are matching to id1 in pedigree 1
        po1 : PedigreeObject for pedigree1
        po2 : PedigreeObject for pedigree2
        gt_set1 : all genotyped ids in pedigree 1
        gt_set2 : all genotyped ids in pedigree 2
        matched_id_dict_1_2 : dict of the form {id1 : id2} mapping ids in pedigree 1 to their matches in pedigree 2
        matched_id_dict_2_1 : dict of the form {id2 : id1} mapping ids in pedigree 2 to their matches in pedigree 1
        mismatch_set1 : set of unmatchable ids in pedigree 1
        mismatch_set2 : set of unmatchable ids in pedigree 2
        strictness : int (0,1,2) controlling how strict we require relative matches to be (see check_relative_concordance)
    """

    if uid1 in mismatch_set1 or uid2 in mismatch_set2:
        return

    pids1 = po1.up_pedigree_dict[uid1][2:] # parent ids of uid1 in tree 1
    pids2 = po2.up_pedigree_dict[uid2][2:] # parent ids of uid2 in tree 2

    if len(pids1) == 0 or len(pids2) == 0:
        return

    p11,p12 = pids1
    p21,p22 = pids2

    p11_rel_set = set([rel for rel in po1.rel_dict[p11]['rel'] | po1.rel_dict[p11]['anc'] if rel in gt_set1])
    p12_rel_set = set([rel for rel in po1.rel_dict[p12]['rel'] | po1.rel_dict[p12]['anc'] if rel in gt_set1])
    p21_rel_set = set([rel for rel in po2.rel_dict[p21]['rel'] | po2.rel_dict[p21]['anc'] if rel in gt_set2])
    p22_rel_set = set([rel for rel in po2.rel_dict[p22]['rel'] | po2.rel_dict[p22]['anc'] if rel in gt_set2])

    p11_desc_set = set([rel for rel in po1.rel_dict[p11]['desc'] if rel in gt_set1])
    p12_desc_set = set([rel for rel in po1.rel_dict[p12]['desc'] if rel in gt_set1])
    p21_desc_set = set([rel for rel in po2.rel_dict[p21]['desc'] if rel in gt_set2])
    p22_desc_set = set([rel for rel in po2.rel_dict[p22]['desc'] if rel in gt_set2])

    # check relative concordance
    p11_p21_rel_check = check_relative_concordance(p11,p21,po1,po2,gt_set1,gt_set2,p11_rel_set,p21_rel_set,strictness=0)
    p12_p22_rel_check = check_relative_concordance(p12,p22,po1,po2,gt_set1,gt_set2,p12_rel_set,p22_rel_set,strictness=0)
    p12_p21_rel_check = check_relative_concordance(p12,p21,po1,po2,gt_set1,gt_set2,p12_rel_set,p21_rel_set,strictness=0)
    p11_p22_rel_check = check_relative_concordance(p11,p22,po1,po2,gt_set1,gt_set2,p11_rel_set,p22_rel_set,strictness=0)

    # Check that all genotyped descendants putative matches are the same degree of separation from them in both trees
    # and that the relationships among those people are the same in both trees.
    p11_p21_desc_check = check_relative_concordance(p11,p21,po1,po2,gt_set1,gt_set2,p11_desc_set,p21_desc_set,strictness=strictness)
    p12_p22_desc_check = check_relative_concordance(p12,p22,po1,po2,gt_set1,gt_set2,p12_desc_set,p22_desc_set,strictness=strictness)
    p12_p21_desc_check = check_relative_concordance(p12,p21,po1,po2,gt_set1,gt_set2,p12_desc_set,p21_desc_set,strictness=strictness)
    p11_p22_desc_check = check_relative_concordance(p11,p22,po1,po2,gt_set1,gt_set2,p11_desc_set,p22_desc_set,strictness=strictness)

    succeeded = True
    to_match_list = []
    if p11_p21_rel_check and p11_p21_desc_check and p12_p22_rel_check and p12_p22_desc_check:
        succeeded = update_match_dicts(p11, p21, po1, po2, gt_set1, gt_set2, matched_id_dict_1_2, matched_id_dict_2_1, mismatch_set1, mismatch_set2)
        if succeeded:
            to_match_list.append([p11,p21])
        succeeded = update_match_dicts(p12, p22, po1, po2, gt_set1, gt_set2, matched_id_dict_1_2, matched_id_dict_2_1, mismatch_set1, mismatch_set2)
        if succeeded:
            to_match_list.append([p12,p22])
    elif p11_p22_rel_check and p11_p22_desc_check and p12_p21_rel_check and p12_p21_desc_check:
        succeeded = update_match_dicts(p11, p22, po1, po2, gt_set1, gt_set2, matched_id_dict_1_2, matched_id_dict_2_1, mismatch_set1, mismatch_set2)
        if succeeded:
            to_match_list.append([p11,p22])
        succeeded = update_match_dicts(p12, p21, po1, po2, gt_set1, gt_set2, matched_id_dict_1_2, matched_id_dict_2_1, mismatch_set1, mismatch_set2)
        if succeeded:
            to_match_list.append([p12,p21])
    else:
        flag_ancestors(p11,po1,mismatch_set1)
        flag_ancestors(p12,po1,mismatch_set1)
        flag_ancestors(p21,po2,mismatch_set2)
        flag_ancestors(p22,po2,mismatch_set2)

    for id1,id2 in to_match_list:
        match_ancestors(id1, id2, po1, po2, gt_set1, gt_set2, matched_id_dict_1_2, matched_id_dict_2_1, mismatch_set1, mismatch_set2)


def match_trees(
    pedigree1 : Dict[int, Tuple[str, int, int]], 
    pedigree2 : Dict[int, Tuple[str, int, int]], 
) -> Dict[int, int]:
    """
    Match ids in normed pedigree1 to ids in normed pedigree2
    Args:
        pedigree1 : {genotype_id : (sex, parent_id1, parent_id2)}
        pedigree2 : {genotype_id : (sex, parent_id1, parent_id2)}
    """
    up_dict1 = {key: [info[0],None,info[1],info[2]] for key,info in pedigree1.items()}
    up_dict2 = {key: [info[0],None,info[1],info[2]] for key,info in pedigree2.items()}
    gt_set1 = {key for key in pedigree1.keys() if not (isinstance(key,int) and key < 0)}
    gt_set2 = {key for key in pedigree2.keys() if not (isinstance(key,int) and key < 0)}

    po1 = PedigreeObject(up_dict1)
    po2 = PedigreeObject(up_dict2)

    overlap_genotyped_id_set = gt_set1 & gt_set2

    matched_id_dict_1_2 = {uid : uid for uid in overlap_genotyped_id_set}
    matched_id_dict_2_1 = {uid : uid for uid in overlap_genotyped_id_set}

    mismatch_set1 = set()
    mismatch_set2 = set()

    for uid in overlap_genotyped_id_set:
        match_ancestors(uid, uid, po1, po2, gt_set1, gt_set2, matched_id_dict_1_2=matched_id_dict_1_2, matched_id_dict_2_1=matched_id_dict_2_1, mismatch_set1=mismatch_set1, mismatch_set2=mismatch_set2)

    return matched_id_dict_1_2


def resolve_age_discrepancies(
    age_dict : Dict[int, int],
    sex_dict : Dict[int, str],
    pairwise_relationships : Dict[int,  Dict[int,  Tuple[Any, Any, Any]]],
    pairwise_log_likelihoods : Dict[int,  Dict[int,  Dict[Tuple[Any, Any, Any],  float]]],
    rel_list : List[int],
    ibd_stats : Dict[FrozenSet[int],Dict[str,Any]],
    estimator : Any
) -> Set[int]:
    """
    WARNING: modifies 
        age_dict
        pairwise_relationships
        pairwise_log_likelihoods

    Find individuals who have two inferred parents of the same sex. This is indicative of mis-specified ages because
    this usually arises in cases where someone has specified their age wrong and they have two children of the same sex,
    or one child of the same sex as their true parent.
    Set all ages of people involved to None and recompute their pairwise relationships and likelihoods.
    Pass and return by reference.

    Args:
        age_dict : {genotype_id : age}
        sex_dict : {genotype_id : sex} (sex is 'M' or 'F')
        pairwise_relationships : {genotype_id1 : {genotype_id2 : deg}}, where deg = (up, down, num_ancs)
        pairwise_log_likelihoods : {genotype_id1 : {genotype_id2 : ll_dict}}, where ll_dict = {(up, down, num_ancs) : log_like}
        rel_list : list of the form [id1, id2, ...] of related genotyped ids
        ibd_stats : Dict mapping pairs frozenset({id1,id2}) to summmary stats of IBD shared between them
        estimator : Instance of RelationshipEstimator()
    """

    def is_twin(id1,id2):
        ibd_info = ibd_stats[frozenset({id1,id2})]
        return estimator.is_twin_pair(ibd_info['total_half'],
                                         ibd_info['total_full'],
                                         ibd_info['num_half'],
                                         age_dict[id1],
                                         age_dict[id2],
                                         sex_dict[id1],
                                         sex_dict[id2])

    age_discrepant_set = set()
    for uid,age in age_dict.items():
        male_parent_set = set()
        female_parent_set = set()
        for rel_id, rel_deg in pairwise_relationships[uid].items():
            if rel_deg == (1,0,1) and (not is_twin(uid,rel_id)):
                rel_sex = sex_dict.get(rel_id,None)
                if rel_sex == 'M' and not np.any([is_twin(rel_id,pid) for pid in male_parent_set]):
                    male_parent_set.add(rel_id)
                elif rel_sex == 'F' and not np.any([is_twin(rel_id,pid) for pid in female_parent_set]):
                    female_parent_set.add(rel_id)
        if len(male_parent_set) > 1 or len(female_parent_set) > 1:
            age_dict[uid] = None
            age_discrepant_set.add(uid)
            for pid in male_parent_set | female_parent_set:
                age_dict[pid] = None
                age_discrepant_set.add(pid)

    # Recompute pairwise_relationships and pairwise_log_likelihoods with ages set to None
    pairwise_relationships, pairwise_log_likelihoods = estimator.get_pairwise_log_likes(
        rel_list = rel_list,
        ibd_stats = ibd_stats,
        age_dict = age_dict,
    )

    return age_discrepant_set


def replace_validated_nodes(
    focal_id : int,
    ped_obj : Any,
    seed_pedigree_list : List[Dict[int, Tuple[str, int, int, int]]],
    validated_node_set_list : List[Set[int]],
    age_dict : Dict[int, int],
    sex_dict : Dict[int, str],
) -> Tuple[Any, Set[int], Set[int], Dict[int, int], Dict[int, int]]:
    """
    Cycle over unplaced validated nodes and try to put them back on the tree.
    if we can't put them back, add them to unplaced_validated_node_set

    Args:
        focal_id : (int)
        ped_obj : Instance of PedigreeObject() for focal_id
        seed_pedigree_list : List of prior pedigrees the form [normed_pedigree1, pedigree2, ...]
        validated_node_set_list : List of the form [set1, set2, ...] of genotyped individuals in seed_pedigree_list
        age_dict : {genotype_id : age}
        sex_dict : {genotype_id : sex}
    """
    new_pedigree = normalize_pedigree(focal_id, ped_obj, ped_obj.up_pedigree_dict, sex_dict, {})
    new_up_dict = build_up_pedigree_dict_from(new_pedigree, age_dict)
    new_ped_obj = PedigreeObject(up_pedigree_dict=new_up_dict)
    unplaced_validated_node_set = set()
    total_placed_validated_node_set = set()
    for old_pedigree,validated_node_set in zip(seed_pedigree_list,validated_node_set_list):
        old_up_dict = build_up_pedigree_dict_from(old_pedigree, age_dict)
        old_ped_obj = PedigreeObject(old_up_dict)
        #old_ped_obj.keep_nodes(validated_node_set,include_parents=True) # yields same pedigree object as pedigree_builder.trim_seed_pedigree
        old_up_dict = old_ped_obj.up_pedigree_dict
        old_down_dict = old_ped_obj.down_pedigree_dict
        #old_pedigree = normalize_pedigree(focal_id, old_ped_obj, old_ped_obj.up_pedigree_dict, sex_dict, {})
        matched_id_dict_new_to_old = match_trees(new_pedigree, old_pedigree)
        matched_id_dict_old_to_new = {val : key for key,val in matched_id_dict_new_to_old.items()}
        #matched_id_dict_new_to_old = {key : val for key,val in matched_id_dict_new_to_old.items() if val in validated_node_set} # restrict to validated nodes
        placed_validated_node_set = {None} # dummy value to get while loop going
        all_placed_id_set = set()
        ct = 0
        while placed_validated_node_set: # if we placed some people and removed them from validated_node_set
            # cycle over matched IDs. Add validated parents if person is founder, add validated children if person is leaf
            new_match_dict_entries = dict()
            placed_validated_node_set = set()
            for new_id,old_id in matched_id_dict_new_to_old.items():
                if old_id in old_up_dict:
                    old_ped_parent_id_set = set(old_pedigree[old_id][1:])
                    validated_parent_id_set = old_ped_parent_id_set & validated_node_set
                    if new_id in new_ped_obj.up_pedigree_dict:
                        new_ped_parent_ids = new_ped_obj.up_pedigree_dict[new_id][2:]
                    else:
                        new_ped_parent_ids = []
                    for pid in validated_parent_id_set - all_placed_id_set:
                        if pid in matched_id_dict_old_to_new and matched_id_dict_old_to_new[pid] in new_ped_parent_ids: # if the person has been placed
                            continue
                        elif np.any([(new_par_id not in matched_id_dict_new_to_old) and (new_par_id not in new_match_dict_entries) for new_par_id in new_ped_parent_ids]): # we can't match some parent of new_id back to the seed pedigree
                            continue
                        pid_rels = old_ped_obj.rel_dict[pid]['rel'] | old_ped_obj.rel_dict[pid]['anc']
                        if np.any([(not (isinstance(rel_id,int) and rel_id < 0)) for rel_id in pid_rels]): # If any relative is genotyped, then this isn't an annotated branch, so don't add it on
                            continue
                        new_parent_rev_map_set = set()
                        for new_pid in new_ped_parent_ids:
                            if new_pid in matched_id_dict_new_to_old:
                                new_parent_rev_map_set.add(matched_id_dict_new_to_old[new_pid])
                            if new_pid in new_match_dict_entries:
                                new_parent_rev_map_set.add(new_match_dict_entries[new_pid])
                        if (len(new_ped_parent_ids) < 1) and ((not new_parent_rev_map_set) or (new_parent_rev_map_set & old_ped_parent_id_set)):
                            new_pid = new_ped_obj.add_parent_for_child(child_id=new_id, # have to get a new id for pid in case pid is alread in new_ped_obj as a different person
                                                                   parent_sex=sex_dict.get(pid),
                                                                   parent_age=age_dict.get(pid))
                            new_match_dict_entries[new_pid] = pid
                            placed_validated_node_set.add(pid)
                            all_placed_id_set.add(pid)
                if old_id in old_down_dict:
                    validated_child_id_set = set(old_down_dict[old_id][2:]) & validated_node_set
                    if new_id in new_ped_obj.down_pedigree_dict:
                        new_ped_child_ids = new_ped_obj.down_pedigree_dict[new_id][2:]
                    else:
                        new_ped_child_ids = []
                    for cid in validated_child_id_set - all_placed_id_set:
                        if cid in matched_id_dict_old_to_new and matched_id_dict_old_to_new[cid] in new_ped_child_ids:
                            continue
                        cid_rels = old_ped_obj.rel_dict[cid]['desc']
                        if np.any([(not (isinstance(desc_id,int) and desc_id < 0)) for desc_id in cid_rels]): # If any descendant is genotyped, then this isn't an annotated branch and it's likely a branch that didn't get matched properly
                            continue
                        new_cid = new_ped_obj.add_child_for_parent(parent_id=new_id, # have to get a new id for cid in case cid is alread in new_ped_obj as a different person
                                                               child_sex=sex_dict.get(cid),
                                                               child_age=age_dict.get(cid))
                        old_ped_parent_id_set = set(old_up_dict[cid][2:])
                        other_old_ped_parent_id_set = old_ped_parent_id_set - {old_id}
                        if other_old_ped_parent_id_set:
                            other_old_parent_id = list(other_old_ped_parent_id_set)[0]
                            other_new_parent_id = matched_id_dict_old_to_new.get(other_old_parent_id,None)
                            if other_new_parent_id:
                                new_ped_obj.connect_parent_child(child_id=new_cid, parent_id=other_new_parent_id)
                            else:
                                other_new_parent_id = new_ped_obj.add_parent_for_child(child_id=new_cid)
                            if (other_new_parent_id not in matched_id_dict_new_to_old) and (other_new_parent_id not in new_match_dict_entries):
                                new_match_dict_entries[other_new_parent_id] = other_old_parent_id
                                matched_id_dict_old_to_new[other_old_parent_id] = other_new_parent_id
                                placed_validated_node_set.add(other_old_parent_id)
                        new_match_dict_entries[new_cid] = cid
                        placed_validated_node_set.add(cid)
                        all_placed_id_set.add(cid)
            matched_id_dict_new_to_old.update(new_match_dict_entries) # update matched dict
            matched_id_dict_old_to_new = {val : key for key,val in matched_id_dict_new_to_old.items()}
            validated_node_set -= placed_validated_node_set
            total_placed_validated_node_set |= placed_validated_node_set

        unplaced_validated_node_set |= {uid for uid in validated_node_set - placed_validated_node_set if (not (isinstance(uid,int) and uid < 0)) and matched_id_dict_old_to_new.get(uid,float('inf')) not in new_ped_obj.all_ids}

    return new_ped_obj,unplaced_validated_node_set,total_placed_validated_node_set,matched_id_dict_new_to_old,matched_id_dict_old_to_new


def validate_input(focal_id, ibd_seg_list, bio_info):
    validate_ibd_seg_list(ibd_seg_list)
    validate_bio_info(bio_info)


def validate_ibd_seg_list(ibd_seg_list):
    for idx,seg in enumerate(ibd_seg_list):
        validate_seg(seg,idx)


def validate_seg(seg,idx):
    # [id1, id2, chromosome, start, end, is_full_ibd, seg_cm]
    if not isinstance(seg[0], int):
        raise Exception("ibd_seg_list[{}][0] must be an int".format(idx))
    elif seg[0] < 0:
        raise Exception("ibd_seg_list[{}][0] must be positive".format(idx))
    if not isinstance(seg[1], int):
        raise Exception("ibd_seg_list[{}][1] must be an int".format(idx))
    elif seg[1] < 0:
        raise Exception("ibd_seg_list[{}][1] must be positive".format(idx))
    if not isinstance(seg[2], str):
        raise Exception("ibd_seg_list[{}][2] must be a str ('1', '2', ..., 'X')".format(idx))
    if not (isinstance(seg[3], int) or isinstance(seg[3], float)):
        raise Exception("ibd_seg_list[{}][3] must be an int or a float".format(idx))
    if not (isinstance(seg[4], int) or isinstance(seg[4], float)):
        raise Exception("ibd_seg_list[{}][4] must be an int or a float".format(idx))
    if not isinstance(seg[5], bool):
        raise Exception("ibd_seg_list[{}][5] must be an int or a bool".format(idx))
    if not (isinstance(seg[6], int) or isinstance(seg[6], float)):
        raise Exception("ibd_seg_list[{}][6] must be an int or a float".format(idx))

def validate_bio_info(bio_info):
    for idx,bio in enumerate(bio_info):
        validate_bio(bio,idx)


def validate_bio(bio,idx):
    if {*bio} != {'genotype_id', 'age', 'sex'}:
        raise Exception("bio_info[{}] keys must be 'genotype_id', 'age', and 'sex'".format(idx))
    if not isinstance(bio['genotype_id'], int):
        raise Exception("bio_info[{}]['genotype_id'] must be an int".format(idx))
    else:
        if bio['genotype_id'] < 0:
            raise Exception("bio_info[{}]['genotype_id'] must be positive".format(idx))
    if not (isinstance(bio['age'], int) or isinstance(bio['age'], float)):
        raise Exception("bio_info[{}]['age'] must be int or float".format(idx))
    if not bio['sex'] in {'M','F'}:
        raise Exception("bio_info[{}]['sex'] must be 'M' or 'F'".format(idx))
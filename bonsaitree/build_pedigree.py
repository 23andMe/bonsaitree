from typing import Any, Dict, List, Set, Tuple, FrozenSet, Optional

import copy

from itertools import combinations
from collections import defaultdict

from .copytools import deepcopy
from .pedigree_builder import PedigreeBuilder
from .connect_pedigree_tools import combine_pedigrees, find_closest_pedigrees
from .ibd_tools import get_related_sets, get_ibd_segs_between_sets, merge_ibd_segs, merge_ibd_segs, get_segment_length_list

INF = float('inf')
UNREL_DEG = (INF,INF,None)


def infer_local_pedigrees(
    focal_id : int,
    sex_dict : Dict[int,str],
    age_dict : Dict[int,int],
    pw_rels : Dict[int, Dict[int,Tuple[Any,Any,Any]]],
    pw_log_likes : Dict[int, Dict[int, Dict[Tuple[Any,Any,Any],float]]],
    ibd_stat_dict : Dict[FrozenSet[int], Dict[str,Any]],
    max_radius : float,
    max_add_degree : int,
    min_rel_append_types : int,
    max_rel_append_types : int,
    ped_save_log_like_delta_fraction : float,
    ped_save_log_like_abs_fraction : float,
    disallow_distant_half_rels : bool,
    use_age_info : bool,
    seed_pedigree_list : Optional[List[Dict[int,Tuple[str,int,int]]]] = None,
    validated_node_set_list : Optional[List[Set[int]]] = None,
) -> Tuple[
        Dict[int,Set[int]],
        Dict[int,List[Any]],
        Dict[int,int],
        Dict[int,Any],
]:
    """
    Take a starting individual, infer the pedigree as far as possible, and store all resulting pedigrees.
    From among the unplaced IDs, choose another seed and repeat.
    Terminate when all individuals have been placed or no more people can be placed.

    Args:
        focal_id : int id of the focal individual for whom we are building a pedigree.
        sex_dict : Dict mapping id to sex (sex is a string 'M' or 'F' or None)
        age_dict : Dict mapping id to age
        pw_rels : Dict mapping ID1 to ID1 to a tuples representing their relationship (up,down,num_ancs)
        pw_log_likes : Dict mapping ID1 to ID1 to a dict mapping relationship tuples to their respective point-predicted likelihoods.
        ibd_stat_dict : Dict mapping ID1 to ID2 to IBD summary stats between ID1 and ID2.
        max_radius : integer (or INF) specifying the maximum degree from the seed id to any other individual in each small pedigree
        max_add_degree : maximum (point-predicted) degree between any new individual we are trying to place and their closest relative in the pedigree.
        min_rel_append_types : minimum number of relationship types to explore when adding an individual to their closest relative
        max_rel_append_types : maximum number of relationship types to explore when adding an individual to their closest relative
        ped_save_log_like_delta_fraction : when sorting pedigrees by likelihood, if a pair of two consecutive pedigrees have likelihoods that differ by more than this amount, keep all pedigrees up to the first in the pair and throw out all pedigrees beyond the first in the pair.
        ped_save_log_like_abs_fraction : when sorting pedigrees by likelihood, throw out all pedigrees whose likelihoods are lower than that of the most likely pedigree by this amount
        disallow_distant_half_rels : do not allow the placement of half relatives beyond half sibling, unless there is an intervening genotyped node to corroborate the half relationship
        use_age_info : use age when computing likelihoods and assessing which configurations are possible (e.g., (id1,id2) = (parent,child), (id1,id2) = (child,parent))
        seed_pedigree_list : List of validated pedigrees whose topologies will remain fixed and onto which we will add all other individuals
        validated_node_set_list : List of sets of nodes whose relationship to the focal individual have been validated
    """

    # don't modify arguments
    seed_pedigree_list_copy = seed_pedigree_list
    validated_node_set_list_copy = validated_node_set_list

    # in the while loop, check to see if there are any remaining seed pedigrees. If so, imput the top one into make_draft_pedigree as the seed. If not, set seed_pedigree=None in the input to make_draft_pedigree.
    index_to_gtid_set = dict()
    index_to_ped_obj_list = dict()
    gtid_to_ped_obj_index = dict()
    traces_dict = dict()
    unplaced_ids = {*pw_rels}
    index = 0
    while unplaced_ids:

        new_pw_rels : Dict[int, Dict[int, Tuple[Any,Any,Any]]] = defaultdict(dict)
        for id1,id2 in combinations( unplaced_ids , r=2 ):
            new_pw_rels[id1][id2] = pw_rels[id1].get(id2, UNREL_DEG)
            new_pw_rels[id2][id1] = pw_rels[id2].get(id1, UNREL_DEG)

        if focal_id in unplaced_ids:
            seed_id = focal_id
        else:
            seed_id = [*unplaced_ids][0]

        local_ped_builder = PedigreeBuilder(
            pairwise_relationships = new_pw_rels,
            pairwise_log_likelihoods = pw_log_likes,
            ibd_stat_dict = ibd_stat_dict,
            sex_dict = sex_dict,
            age_dict = age_dict,
            use_age_info = use_age_info,
            min_rel_append_types = min_rel_append_types,
            max_rel_append_types = max_rel_append_types,
        )

        local_ped_builder.enforce_relationships()

        seed_pedigree = None
        validated_node_set = None
        if seed_pedigree_list_copy and validated_node_set_list_copy:
            seed_pedigree = seed_pedigree_list_copy[0]
            validated_node_set = validated_node_set_list_copy[0]
            seed_pedigree_list_copy = seed_pedigree_list[1:]
            validated_node_set_list_copy = validated_node_set_list_copy[1:]

        build_succeeded = local_ped_builder.make_draft_pedigree(
            seed_id = seed_id,
            max_radius = max_radius,
            max_add_degree = max_add_degree,
            ped_save_log_like_delta_fraction = ped_save_log_like_delta_fraction,
            ped_save_log_like_abs_fraction = ped_save_log_like_abs_fraction,
            seed_pedigree = seed_pedigree,
            validated_node_set = validated_node_set,
            disallow_distant_half_rels = disallow_distant_half_rels
        )

        traces_dict[index] = local_ped_builder.traces

        placed_ids = unplaced_ids - set(local_ped_builder.unplaced_ids)

        index_to_gtid_set[index] = placed_ids
        for gtid in list(placed_ids):
            gtid_to_ped_obj_index[gtid] = index

        ped_obj_list = [info[0] for info in local_ped_builder.ped_obj_list]
        for ped_obj in ped_obj_list:
            ped_obj.update_all_rels()
        ped_obj_list = sorted(ped_obj_list, key = lambda x: x.pedigree_log_likelihood, reverse=True)

        index_to_ped_obj_list[index] = ped_obj_list
        unplaced_ids = {*local_ped_builder.unplaced_ids}

        index += 1

    return (index_to_gtid_set, index_to_ped_obj_list, gtid_to_ped_obj_index, traces_dict)


def assemble_local_pedigrees(
    focal_id : int,
    ibd_seg_list : List[List[Any]],
    ibd_stat_dict : Dict[FrozenSet[int], Dict[str,Any]],
    pw_rels : Dict[int, Dict[int,Tuple[Any,Any,Any]]],
    pw_log_likes : Dict[int, Dict[int, Dict[Tuple[Any,Any,Any],float]]],
    index_to_gtid_set : Dict[int, Set[int]],
    index_to_ped_obj_list : Dict[int, Any],
    gtid_to_ped_obj_index : Dict[int, int],
    num_ped_objs : int,
    disallow_distant_half_rels : bool,
    drop_ibd_alpha : float,
) -> Tuple[
        List[Any],
        Dict[int,Any],
        Dict[int,Set[int]],
        List[Tuple[int,int,bool]],
    ]:

    """
    Take locally inferred pedigrees and assemble them ito a big pedigree.

    Args:
        focal_id : int id of the focal individual for whom we are building a pedigree.
        ibd_seg_list : list of the form [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
        ibd_stat_dict : Dict mapping ID1 to ID2 to IBD summary stats between ID1 and ID2.
        pw_rels : Dict mapping ID1 to ID1 to a tuples representing their relationship (up,down,num_ancs)
        pw_log_likes : Dict mapping ID1 to ID1 to a dict mapping relationship tuples to their respective point-predicted likelihoods.
        index_to_ped_obj_list : Dict mapping an index to each pedigree object
        index_to_gtid_set : Dict mapping the index of the pedigree object to the set of genotyped IDs in the object
        gtid_to_ped_obj_index : Dict mapping genotyped node id to the index of the corresponding pedigree object in which the individual was placed.
    """

    # don't modify arguments
    index_to_gtid_set_copy = copy.deepcopy(index_to_gtid_set)
    index_to_ped_obj_list_copy = copy.deepcopy(index_to_ped_obj_list)
    gtid_to_ped_obj_index_copy = copy.deepcopy(gtid_to_ped_obj_index)

    # combine pedigrees
    combine_trace = []
    stop = False
    while len(index_to_ped_obj_list_copy) > 1:
        ped_idxs = find_closest_pedigrees(
            index_to_gtid_set=index_to_gtid_set_copy,
            ibd_seg_list=ibd_seg_list,
        )
        if ped_idxs:  # In the future, if we don't pull everyone as related to the focal ID, the remaining peds can share no IBD.
            idx1,idx2 = ped_idxs
        else:
            stop = True
            break
        ped_obj_list1 = index_to_ped_obj_list_copy[idx1]
        ped_obj_list2 = index_to_ped_obj_list_copy[idx2]
        all_gt_set1 = index_to_gtid_set_copy[idx1]
        all_gt_set2 = index_to_gtid_set_copy[idx2]

        new_ped_obj_list : List[Any] = []
        po1_po2_log_like_list = [(po1, po2, po1.pedigree_log_likelihood + po2.pedigree_log_likelihood) for po1 in ped_obj_list1 for po2 in ped_obj_list2]
        po1_po2_log_like_list = sorted(po1_po2_log_like_list, key = lambda x: -x[2])

        for po1,po2,_ in po1_po2_log_like_list[:num_ped_objs]:
            po1_copy = deepcopy(po1)
            po2_copy = deepcopy(po2)

            po1_copy.keep_nodes(keep_gt_node_set=all_gt_set1, include_parents=False) # remove any unnecessary ancestors
            po2_copy.keep_nodes(keep_gt_node_set=all_gt_set2, include_parents=False)

            min_ind1 = po1_copy.min_parent_ind
            min_ind2 = po2_copy.min_parent_ind
            min_ind = min(min_ind1,min_ind2)
            po2_copy.update_ungenotyped_inds(min_ind-1)

            combined_ped_obj_list = combine_pedigrees(
                po1 = po1_copy,
                po2 = po2_copy,
                ibd_seg_list = ibd_seg_list,
                pw_rels = pw_rels,
                ibd_stat_dict = ibd_stat_dict,
                pw_log_likes = pw_log_likes,
                disallow_distant_half_rels = disallow_distant_half_rels,
                drop_ibd_alpha = drop_ibd_alpha,
            )
            new_ped_obj_list += combined_ped_obj_list

            if combined_ped_obj_list: # If we were able to combine the pedigrees, don't try more
                break

        keep_idx = idx1
        del_idx = idx2
        ped_obj_list = sorted(new_ped_obj_list, key = lambda ped_obj: -ped_obj.pedigree_log_likelihood)
        id_set = all_gt_set1 | all_gt_set2
        if not new_ped_obj_list:
            if focal_id in all_gt_set1:
                keep_idx = idx1
                del_idx = idx2
                ped_obj_list = ped_obj_list1
                id_set = all_gt_set1
            elif focal_id in all_gt_set2:
                keep_idx = idx2
                del_idx = idx1
                ped_obj_list = ped_obj_list2
                id_set = all_gt_set2
            else:
                focal_id_idx = gtid_to_ped_obj_index_copy[focal_id]
                focal_all_gt_set = index_to_gtid_set_copy[focal_id_idx]

                chrom_ibd_segs_dict = get_ibd_segs_between_sets(focal_all_gt_set, all_gt_set1, ibd_seg_list)
                merged_chrom_ibd_segs_dict = merge_ibd_segs(chrom_ibd_segs_dict)
                merged_ibd_seg_lengths = get_segment_length_list(merged_chrom_ibd_segs_dict)
                L_tot1 = sum(merged_ibd_seg_lengths)

                chrom_ibd_segs_dict = get_ibd_segs_between_sets(focal_all_gt_set, all_gt_set2, ibd_seg_list)
                merged_chrom_ibd_segs_dict = merge_ibd_segs(chrom_ibd_segs_dict)
                merged_ibd_seg_lengths = get_segment_length_list(merged_chrom_ibd_segs_dict)
                L_tot2 = sum(merged_ibd_seg_lengths)

                if L_tot1 > L_tot2:
                    keep_idx = idx1
                    del_idx = idx2
                    ped_obj_list = ped_obj_list1
                    id_set = all_gt_set1
                else:
                    keep_idx = idx2
                    del_idx = idx2
                    ped_obj_list = ped_obj_list2
                    id_set = all_gt_set2

        if new_ped_obj_list:
            combine_trace.append([keep_idx, del_idx, True])
        else:
            combine_trace.append([keep_idx, del_idx, False])

        index_to_gtid_set_copy[keep_idx] = id_set
        index_to_ped_obj_list_copy[keep_idx] = ped_obj_list
        for uid in index_to_gtid_set_copy[del_idx]:
            del gtid_to_ped_obj_index_copy[uid]
        for uid in id_set:
            gtid_to_ped_obj_index_copy[uid] = keep_idx
        index_to_ped_obj_list_copy.pop(del_idx)
        index_to_gtid_set_copy.pop(del_idx)

    keep_idx = [*index_to_gtid_set_copy][0]
    all_id_set = index_to_gtid_set_copy[keep_idx]
    ped_obj_list = index_to_ped_obj_list_copy[keep_idx]
    ped_obj = ped_obj_list[0]

    focal_po_index = gtid_to_ped_obj_index_copy[focal_id]
    ped_obj_list = index_to_ped_obj_list_copy[focal_po_index]

    return ped_obj_list, index_to_ped_obj_list_copy, index_to_gtid_set_copy, combine_trace


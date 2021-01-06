from typing import Any, Dict, List, Set, Tuple
import os
import json

from scipy import interpolate
from funcy import memoize

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
GENETIC_MAP_FILE = os.path.join(MODELS_DIR, 'ibd64_metadata_dict.json')



@memoize
def get_map_interpolator() -> Dict[str, Any]:
    """
    Make a dictionary of interpolators. map_interpolator[chrom] = interpolator,
    where interpolator(phys_posit) gives the genetic position of phys_posit.
    """
    metadata_dict = json.loads(open(GENETIC_MAP_FILE).read())
    map_interpolator = dict()
    for chrom,posits in metadata_dict.items():
        phys_posits = [pos[0] for pos in posits]
        genet_posits = [pos[1] for pos in posits]
        interpolator = interpolate.interp1d(phys_posits,genet_posits)
        map_interpolator[chrom] = interpolator
    return map_interpolator


@memoize
def get_reverse_map_interpolator() -> Dict[str, Any]:
    """
    Make a dictionary of interpolators. map_interpolator[chrom] = interpolator,
    where interpolator(phys_posit) gives the genetic position of phys_posit.
    """
    metadata_dict = json.loads(open(GENETIC_MAP_FILE).read())
    map_interpolator = dict()
    for chrom,posits in metadata_dict.items():
        phys_posits = [pos[0] for pos in posits]
        genet_posits = [pos[1] for pos in posits]
        interpolator = interpolate.interp1d(genet_posits,phys_posits)
        map_interpolator[chrom] = interpolator
    return map_interpolator


def get_related_sets(
    id_set1 : Set[int],
    id_set2 : Set[int],
    ibd_seg_list : List[List[Any]],
) -> Tuple[Set[int],Set[int]]:
    """
    Find the individuals in id_set1 who share IBD with individuals in id_set2 and vice versa.
    Args:
        id_set1: First set of ids we want to subset
        id_set2: Second set of ids we want to subset
        ibd_seg_list: list of the form [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
    """
    rels1 = set()
    rels2 = set()
    for id1, id2, chrom, start, end, is_full, cm in ibd_seg_list:
        if (id1 in id_set1) and (id2 in id_set2):
            if cm > 0:
                rels1.add(id1)
                rels2.add(id2)
        elif (id2 in id_set1) and (id1 in id_set2):
            if cm > 0:
                rels1.add(id2)
                rels2.add(id1)
    return (rels1,rels2)


def get_ibd_segs_between_sets(
    leaves1 : Set[int],
    leaves2 : Set[int],
    ibd_seg_list : List[List[Any]]
) -> Dict[str, List[Tuple[Any,Any]]]:
    """
    Get IBD segments between leaves1 and leaves2 and store them in a dict with chromosomes as keys
    Args:
        leaves1: First set of leaves
        leaves2: Second set of leaves
        ibd_seg_list: list of the form [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]
    """
    chrom_ibd_segs_dict : Dict[str, List[Tuple[Any,Any]]] = dict() # Dict of the form { chrom : [[start,end],[start,end],...] }
    for id1, id2, chrom, start, end, is_full, cm in ibd_seg_list:
        if (id1 in leaves1 and id2 in leaves2) or (id1 in leaves2 and id2 in leaves1):
            if chrom not in chrom_ibd_segs_dict:
                chrom_ibd_segs_dict[chrom] = []
            chrom_ibd_segs_dict[chrom].append((start,end))
    return chrom_ibd_segs_dict


def merge_ibd_segs(
    chrom_ibd_segs_dict: Dict[str, List[Tuple[Any,Any]]]
) -> Dict[str, List[Tuple[Any,Any]]]:
    """
    Sort and merge segments.
    Args:
        chrom_ibd_segs_dict: Dict mapping chromosome indicator (as a string such as '1' or 'X') to a list of 
            (start,end) positions for segments found on that chromosome.
    """
    merged_chrom_ibd_segs_dict = dict()
    for chrom,seg_list in chrom_ibd_segs_dict.items():
        ibd_segs = sorted(seg_list,key=lambda x: (x[0],x[1]))
        merged_ibd = [ibd_segs[0]]
        for seg in ibd_segs[1:]:
            prev_seg = merged_ibd[-1]
            if seg[0] <= prev_seg[1]: # If the start of the next segment is before the end of the segment before it
                new_seg_start = merged_ibd[-1][0]
                new_seg_end = max(seg[1],prev_seg[1])
                new_seg = (new_seg_start,new_seg_end)
                merged_ibd[-1] = new_seg
            else:
                merged_ibd.append(seg)
        merged_chrom_ibd_segs_dict[chrom] = merged_ibd
    return merged_chrom_ibd_segs_dict


def get_segment_length_list(
    chrom_ibd_segs_dict : Dict[str, List[Tuple[Any,Any]]],
):
    """
    Get a list containing the length of each segment.
    Args:
        chrom_ibd_segs_dict: Dict mapping chromosome indicator (as a string such as '1' or 'X') to a list of 
            (start,end) positions for segments found on that chromosome.
    """
    map_interpolator = get_map_interpolator()
    seg_len_list = []
    for chrom,segs in chrom_ibd_segs_dict.items():
        for start,end in segs:
            try:
                start_genet = map_interpolator[chrom](start)
            except:
                if start < map_interpolator[chrom].x[0]: # If it's off the beginning, set genetic position to first genetic position
                    start_genet = map_interpolator[chrom].y[0]
                else: # Then the start is probably off the end so we should not consider this segment
                    continue
            try:
                end_genet = map_interpolator[chrom](end)
            except:
                if end > map_interpolator[chrom].x[-1]: # If it's off the end, set end genetic position to last genetic position
                    end_genet = map_interpolator[chrom].y[-1]
                else:
                    continue
            genet_length = end_genet - start_genet
            seg_len_list.append(genet_length)
    return seg_len_list


def check_overlap(
    focal_id_set : Set[int],
    rel_id_set1 : Set[int],
    rel_id_set2 : Set[int],
    ibd_seg_list : List[List[Any]],
    threshold : float = 0.05,
) -> bool:
    """
    Check whether segments between focal_id_set and rel_id_set1 overlap
    with segments between focal_id_set and rel_id_set2. If so, then rel_id_set1
    and rel_id_set2 cannot be related to focal_id_set through the same ancestor
    of focal_id_set.
    Args:
        focal_id_set: Set of IDs for which two ancestral branches may be placed on the wrong side
        rel_id_set1: leaves of one ancestral branch
        rel_id_set2: leaves of another ancestral branch
        threshold: percent overlap of IBD(focal_id_set,rel_id_set1) segments and
                   IBD(focal_id_set,rel_id_set2) segments for us to say that the
                   overlap is inconsistent with rel_id_set1 and rel_id_set1 being
                   placed on ancestral branches that stem from an ancestral spouse pair.
    """

    # Get IBD segments between the focal_id_set and rel_id_set1
    chrom_ibd_segs_dict1 = dict()
    chrom_ibd_segs_dict1 = get_ibd_segs_between_sets(focal_id_set, rel_id_set1, ibd_seg_list) 
    merged_chrom_ibd_segs_dict1 = merge_ibd_segs(chrom_ibd_segs_dict1) # Sort and merge segments
    seg_len_list1 = get_segment_length_list(merged_chrom_ibd_segs_dict1)
    L_merged_tot1 = sum(seg_len_list1)

    # Get IBD segments between the focal_id_set and rel_id_set2
    chrom_ibd_segs_dict2 = dict()
    chrom_ibd_segs_dict2 = get_ibd_segs_between_sets(focal_id_set, rel_id_set2, ibd_seg_list) 
    merged_chrom_ibd_segs_dict2 = merge_ibd_segs(chrom_ibd_segs_dict2) # Sort and merge segments
    seg_len_list2 = get_segment_length_list(merged_chrom_ibd_segs_dict2)
    L_merged_tot2 = sum(seg_len_list2)

    # merge segments beteween focal and both other sets
    all_seg_dict = merged_chrom_ibd_segs_dict1
    for chrom,seg_list in merged_chrom_ibd_segs_dict2.items():
        if chrom not in all_seg_dict:
            all_seg_dict[chrom] = seg_list
        else:
            all_seg_dict[chrom] += seg_list
    merged_chrom_ibd_segs_dict_all = merge_ibd_segs(all_seg_dict)
    merged_seg_len_list = get_segment_length_list(merged_chrom_ibd_segs_dict_all)
    L_merged_tot_all = sum(merged_seg_len_list)

    L_overlap = L_merged_tot1 + L_merged_tot2 - L_merged_tot_all # overlap between merged_chrom_ibd_segs_dict1 and merged_chrom_ibd_segs_dict2
    is_overlap = False
    if L_overlap / L_merged_tot_all >= threshold:
        is_overlap = True

    return is_overlap

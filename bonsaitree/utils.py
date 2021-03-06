from typing import Any, DefaultDict, Dict, FrozenSet, List, Set, Tuple

import os
from collections import defaultdict


def transform_segment_lists_to_ibd_summaries(
    segments: List[List[Any]],
) -> Dict[FrozenSet[int], Dict[str, Any]]:
    """
    Converts a list of ibd_segment lists to a dict of ibd_summaries

    Args:
        segments: List of IBD segments to generate summary statistics for.
            Has the following format:
            [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]...]
    """

    ibd_stats: DefaultDict[FrozenSet[int], Dict[str, Any]] = defaultdict(
        lambda: {"total_half": 0, "total_full": 0, "num_half": 0, "max_seg_cm": 0}
    )

    observed_pairs: Set[Tuple[int, int]] = set()
    for s in segments:
        id1, id2, chromosome, start, end, is_full_ibd, seg_cm = s
        if (id2, id1) in observed_pairs:
            raise ValueError("Duplicate IBD Segment data")
        observed_pairs.add((id1, id2))

        key = frozenset([id1, id2])
        ibd_stats[key]["total_half"] += seg_cm if not is_full_ibd else 0
        ibd_stats[key]["total_full"] += seg_cm if is_full_ibd else 0
        ibd_stats[key]["num_half"] += int(not is_full_ibd)
        ibd_stats[key]["max_seg_cm"] = max(ibd_stats[key]["max_seg_cm"], seg_cm)

    return ibd_stats


def transform_segment_dicts_to_ibd_summaries(
    segments: List[Dict[str, Any]]
) -> Dict[FrozenSet[int], Dict[str, Any]]:
    """
    Converts a list of ibd_segment data to a dict of ibd_summaries

    Args:
        segments: List of IBD segments to generate summary statistics for.
        Has the following format: {
            genotype_id_1: int,
            genotype_id_2: int,
            chromosome: str,
            start: int,
            end: int,
            is_full_ibd: bool,
            seg_cm: float
        }
    """

    ibd_stats: DefaultDict[FrozenSet[int], Dict[str, Any]] = defaultdict(
        lambda: {"total_half": 0, "total_full": 0, "num_half": 0, "max_seg_cm": 0}
    )

    observed_pairs: Set[Tuple[int, int]] = set()
    for s in segments:
        if (s["genotype_id_2"], s["genotype_id_1"]) in observed_pairs:
            raise ValueError("Duplicate IBD Segment data")
        observed_pairs.add((s["genotype_id_1"], s["genotype_id_2"]))

        key = frozenset([s["genotype_id_1"], s["genotype_id_2"]])
        ibd_stats[key]["total_half"] += s["seg_cm"] if not s["is_full_ibd"] else 0
        ibd_stats[key]["total_full"] += s["seg_cm"] if s["is_full_ibd"] else 0
        ibd_stats[key]["num_half"] += int(not s["is_full_ibd"])
        ibd_stats[key]["max_seg_cm"] = max(ibd_stats[key]["max_seg_cm"], s["seg_cm"])

    return ibd_stats


def read_ibis_ibd(
    ibis_dir_path: str,
):
    """
    Read Ibis segments for all chromosomes and convert them to Bonsai format.

    Args:
        ibis_dir_path: Path to directory containing .seg files generated by IBIS
                       for each chromosome.
    """

    seg_file_list = [
        f
        for f in os.listdir(ibis_dir_path)
        if f.endswith('.seg')
    ]

    ibd_seg_list = []
    for seg_file in seg_file_list:
        file_path = os.path.join(ibis_dir_path, seg_file)
        raw_ibis_segs = open(file_path).readlines()
        chrom_ibd_seg_list = ibis_to_ibd_seg_list(
            raw_ibis_segs = raw_ibis_segs,
        )
        ibd_seg_list += chrom_ibd_seg_list

    return chrom_ibd_seg_list


def ibis_to_ibd_seg_list(
    raw_ibis_segs: List[str],
) -> List[List[Any]]:
    """
    Converts the output in a .seg file from IBIS (Seidman et al, 2020) to
    segments that can be input to Bonsai. Specifically, creates the "IBD1 or IBD2"
    class of segments by merging IBD1 and IBD2 segments

    Args:
        raw_ibis_segs: lines from the ibis output file for a given chromosome.
    """

    raw_ibis_segs = [
        d.split()
        for d in raw_ibis_segs
    ]

    half_seg_list = [[None] * 9]
    full_seg_list = [[None] * 9]
    for seg in raw_ibis_segs:
        (
            id1, 
            id2, 
            chrom, 
            phys_start, 
            phys_end, 
            ibd_type, 
            gen_start, 
            gen_end, 
            gen_seg_len, 
            num_snps, 
            err_ct, 
            err_density,
        ) = seg

        id1 = int(id1.split(':')[0])
        id2 = int(id2.split(':')[0])
        phys_start = float(phys_start)
        phys_end = float(phys_end)
        gen_start = float(gen_start)
        gen_end = float(gen_end)
        gen_seg_len = float(gen_seg_len)

        # convert 'IBDX' text string to boolean is_full
        is_full = False
        if ibd_type == 'IBD2':
            is_full = True

        # set seg info
        seg_info = [
            id1, 
            id2, 
            chrom, 
            phys_start, 
            phys_end, 
            gen_start, 
            gen_end, 
            is_full, 
            gen_seg_len,
        ]

        # either merge seg with previous half IBD segment or record a new seg
        last_half_seg_info = half_seg_list[-1]
        (
            prev_id1, 
            prev_id2, 
            prev_chrom, 
            prev_phys_start, 
            prev_phys_end, 
            prev_gen_start, 
            prev_gen_end, 
            prev_is_full, 
            prev_gen_seg_len,
        ) = last_half_seg_info

        if id1 == prev_id1 and id2 == prev_id2 and chrom == prev_chrom and (prev_gen_end >= gen_start):
            half_seg_list[-1][4] = phys_end
            half_seg_list[-1][6] = gen_end
            half_seg_list[-1][8] = gen_end - prev_gen_start
        elif not is_full:
            half_seg_list.append(seg_info)

        # either merge seg with previous full IBD segment or record a new seg
        if is_full:
            last_full_seg_info = full_seg_list[-1]
            (
                prev_id1, 
                prev_id2, 
                prev_chrom, 
                prev_phys_start, 
                prev_phys_end, 
                prev_gen_start, 
                prev_gen_end, 
                prev_is_full, 
                prev_gen_seg_len,
            ) = last_full_seg_info
            
            if id1 == prev_id1 and id2 == prev_id2 and chrom == prev_chrom and (prev_gen_end >= gen_start):
                full_seg_list[-1][4] = phys_end
                full_seg_list[-1][6] = gen_end
                full_seg_list[-1][8] = gen_end - prev_gen_start
            else:
                full_seg_list.append(seg_info)

    # strip off "None" tuple from each list
    half_seg_list = half_seg_list[1:]
    full_seg_list = full_seg_list[1:]

    # add segments to list
    ibd_seg_list = []
    for seg_info in half_seg_list + full_seg_list:
        (
            id1, 
            id2, 
            chrom, 
            phys_start, 
            phys_end, 
            gen_start, 
            gen_end, 
            is_full, 
            gen_seg_len,
        ) = seg_info
        seg = (id1, id2, chrom, phys_start, phys_end, is_full, gen_seg_len)
        ibd_seg_list.append(seg)

    return ibd_seg_list
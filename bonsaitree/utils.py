from typing import Any, DefaultDict, Dict, FrozenSet, List, Set, Tuple

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

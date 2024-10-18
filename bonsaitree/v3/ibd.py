import json
from collections import defaultdict
from typing import Any

from funcy import memoize
from scipy import interpolate

from .constants import GENETIC_MAP_FILE


@memoize
def get_map_interpolator() -> dict[str, Any]:
    """
    Make a dictionary of interpolators.

    Args:
        None

    Returns:
        map_interpolator: Dict mapping chromosome indicator (as
            a string such as '1' or 'X') to an interpolator.
            where interpolator(phys_posit) gives the genetic
            position of phys_posit.
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
def get_reverse_map_interpolator() -> dict[str, Any]:
    """
    Make a dictionary of interpolators.

    Args:
        None

    Returns:
        map_interpolator: dict
            Dict mapping chromosome indicator (as a string such as
            '1' or 'X') to an interpolator. where interpolator(genet_posit)
            gives the physical position of genet_posit.
    """
    metadata_dict = json.loads(open(GENETIC_MAP_FILE).read())
    map_interpolator = dict()
    for chrom,posits in metadata_dict.items():
        phys_posits = [pos[0] for pos in posits]
        genet_posits = [pos[1] for pos in posits]
        interpolator = interpolate.interp1d(genet_posits,phys_posits)
        map_interpolator[chrom] = interpolator
    return map_interpolator


def phys_to_genet(
    pos: float,
    chrom: str,
)-> float:
    """
    Convert a physical position to a genetic position.

    Args:
        pos: physical position
        chrom: chromosome

    Returns:
        genet_pos: genetic position
    """
    chrom = str(chrom)  # ensure it's a string
    map_interpolator = get_map_interpolator()
    if pos < map_interpolator[chrom].x[0]:
        return map_interpolator[chrom].y[0]
    elif pos > map_interpolator[chrom].x[-1]:
        return map_interpolator[chrom].y[-1]
    else:
        return map_interpolator[chrom](pos)


def genet_to_phys(
    pos: float,
    chrom: str,
)-> float:
    """
    Convert a genetic position to a physical position.

    Args:
        pos: genetic position
        chrom: chromosome

    Returns:
        phys_pos: physical position
    """
    map_interpolator = get_reverse_map_interpolator()
    if pos < map_interpolator[chrom].x[0]:
        return map_interpolator[chrom].y[0]
    elif pos > map_interpolator[chrom].x[-1]:
        return map_interpolator[chrom].y[-1]
    else:
        return map_interpolator[chrom](pos)


def get_pair_to_seg_dict(
    phased_ibd_seg_list: list[list[int]],
):
    """
    Map each pair of IDs to the set of their shared segments.

    Args:
        phased_ibd_seg_list: List of the form
            [[id1, id2, hap1, hap2, chrom_str, start, end, seg_len_cm],...]
    Returns:
        seg_dict: Dict of the form
            {
                frozenset({i1, i2}) :
                    [[id1, id2, hap1, hap2, chrom_str, start, end, seg_len_cm],...]
            ,...}
    """
    seg_dict = {}
    for seg in phased_ibd_seg_list:
        i1, i2, h1, h2, chrom_str, start, end, seg_len = seg
        pair = frozenset({i1, i2})
        if pair not in seg_dict:
            seg_dict[pair] = []
        seg_dict[pair].append(seg)
    return seg_dict


def get_chrom_to_seg_list(
    phased_ibd_seg_list: list[list[int]],
):
    """
    Take a bunch of segments and return a dict
    mapping each chromosome to the list of segments
    on that chromosome.

    Args:
        seg_list: list
            List of the form
             [[id1, id2, hap1, hap2, chrom_str, gen_start, gen_end, seg_len_cm],...]

    Returns:
        chrom_to_seg_list: Dict of the form {chrom: seg_list, ...}
    """

    phased_ibd_seg_list = sorted(phased_ibd_seg_list, key=lambda x: (x[4], x[5], x[6]))

    chrom_to_seg_list = {}
    for seg in phased_ibd_seg_list:
        i1, i2, h1, h2, chrom, start, end, seg_len = seg

        if chrom not in chrom_to_seg_list:
            chrom_to_seg_list[chrom] = []

        chrom_to_seg_list[chrom].append(seg)

    return chrom_to_seg_list


def get_unphased_to_phased(
    unphased_ibd_seg_list: list[tuple[int,int,str,float,float,bool,float]],
):
    """
    Convert unphased IBD segments to phased IBD segments.
    Clearly, this can't handle haplotype assignment; so we
    just assign haplotype 0,0 to all segments except when
    there is a full IBD segment. In that case, we assign
    one segment to haplotype 0,0 and the other to haplotype 1,1.

    Args:
        ibd_seg_list: List of the form [[id1, id2, chromosome, start_bp, end_bp, is_full_ibd, seg_len_cm],...]

    Returns:
        phased_ibd_seg_list: List of the form [[id1, id2, hap1, hap2, chromosome, start_cm, end_cm, cm],...]
    """  # noqa: E501
    phased_ibd_seg_list = []
    for id1, id2, chrom, start_bp, end_bp, _, _ in unphased_ibd_seg_list:
        start_cm = phys_to_genet(start_bp, chrom)
        end_cm = phys_to_genet(end_bp, chrom)
        seg_len_cm = end_cm - start_cm
        phased_ibd_seg_list.append([id1, id2, 0, 0, chrom, start_cm, end_cm, seg_len_cm])
    return phased_ibd_seg_list


def get_phased_to_unphased(
    phased_ibd_seg_list: list[list[int]],
):
    """
    Convert phased IBD segments to IBD64-style segments

    Args:
        ibd_seg_list: list
            List of the form
                [[id1, id2, hap1, hap2, chrom_str, gen_start, gen_end, seg_len_cm],...]

    Returns:
        seg_list: list
            List of the form
                [[id1, id2, chromosome, start_bp, end_bp, is_full_ibd, seg_len_cm],...]
    """
    seg_dict = get_pair_to_seg_dict(phased_ibd_seg_list)

    # get the reverse map interpolator
    interp = get_reverse_map_interpolator()

    new_seg_list = []
    for pair,all_seg_list in seg_dict.items():
        sorted_seg_list = sorted(all_seg_list, key=lambda x: (x[4], x[5], x[6]))
        chrom_to_seg_dict = get_chrom_to_seg_list(sorted_seg_list)

        for chrom, seg_list in chrom_to_seg_dict.items():

            interval_list = [[s[5], s[6]] for s in seg_list]

            # compute half and full IBD
            half_seg_list = get_merger(interval_list)
            full_seg_list = get_overlap(interval_list)
            full_seg_list = get_merger(full_seg_list)

            # ensure chrom is a string
            chrom_str = str(chrom)

            for start_cm,end_cm in half_seg_list:
                seg_len_cm = end_cm - start_cm
                start_bp = float(interp[chrom_str](start_cm))
                end_bp = float(interp[chrom_str](end_cm))
                record_seg = [*pair] + [chrom, start_bp, end_bp, False, seg_len_cm]
                new_seg_list.append(record_seg)

            for start_cm,end_cm in full_seg_list:
                seg_len_cm = end_cm - start_cm
                start_bp = float(interp[chrom_str](start_cm))
                end_bp = float(interp[chrom_str](end_cm))
                record_seg = [*pair] + [chrom, start_bp, end_bp, True, seg_len_cm]
                new_seg_list.append(record_seg)

    return new_seg_list


def get_intersect(
    segs1: list[list[int]],
    segs2: list[list[int]],
):
    """
    Get the intersection between two
    sets of segments.

    Args:
        segs1: list of the form [[start,end],...]
            assume segments are disjoint and
            sorted from left to right position
        segs2: list of the form [[start,end],...]
            assume segments are disjoint and
            sorted from left to right position

    Returns:
        intersection: list of the form [[start,end],...]
            containing segments that overlap between
            segs1 and segs2.
    """
    n1 = len(segs1)
    n2 = len(segs2)
    i1 = i2 = 0
    intersection = []
    while i1 < n1 and i2 < n2:
        s1,e1 = segs1[i1]
        s2,e2 = segs2[i2]

        if e1 < s2:
            i1 += 1
        elif e2 < s1:
            i2 += 1
        else:
            start = max(s1, s2)
            end = min(e1, e2)

            intersection.append([start, end])

            if e1 < e2:
                i1 += 1
            else:
                i2 += 1

    return intersection


def get_overlap(
    seg_list: list[list[int]],
):
    """
    Get any parts where the segments in a single
    seg list overlap with one another.

    Args:
        seg_list: list of the form [[start,end],...]

    Returns:
        overlap: list of the form [[start,end],...]
    """

    seg_list = sorted(seg_list)  # ensure the segments are sorted
    merged = []
    overlap = []
    for start, end in seg_list:
        if merged and start < merged[-1][1]:
            ol_start = max(start, merged[-1][0])
            ol_end = min(end, merged[-1][1])
            overlap.append([ol_start, ol_end])
        if len(merged) == 0 or merged[-1][1] < start:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return overlap


def get_merger(
    seg_list: list[list[int]],
):
    """
    Get the merger of two sets of [start, end] regions. Any overlapping
    segments are merged into the same segment.

    Args:
        seg_list: list of the form [[start,end],...]
            assume segments sorted from left to right
            position.

    Returns:
        merged: list of the form [[start,end],...]
            in which overlapping segments have been
            merged.
    """
    seg_list = sorted(seg_list)  # ensure the segments are sorted
    merged = []
    for start, end in seg_list:
        if len(merged) == 0 or merged[-1][1] < start:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def get_segs_between_sets(
    id_set1: set[int],
    id_set2: set[int],
    ibd_seg_list: list[list[int]],
):
    """
    Get all segments between two sets of IDs

    Args:
        id_set1: set of genotyped IDs
        id_set2: set of genotyped IDs
        ibd_seg_list: List of segments of the form
            [[id1, id2, hap1, hap2, chrom, start, end, cm]]

    Returns:
        ibd_segs: list of segments with one ID in id_set1 and one
                  ID in id_set2
    """
    return [
        s
        for s in ibd_seg_list
        if (s[0] in id_set1 and s[1] in id_set2)
        or (s[1] in id_set1 and s[0] in id_set2)
    ]


def get_id_to_shared_ibd(
    id_set1: set[int],
    id_set2: set[int],
    ibd_seg_list: list[list[int]],
):
    """
    Get a dict of the form {i : L,...} mapping
    each ID in id_set1 to the total length they
    share with all people in id_set2.

    Args:
        id_set1: set of IDs from one pedigree
        id_set2: set of IDs from another pedigree
        ibd_seg_list: List of segments of the form
            [[id1, id2, hap1, hap2, chrom, start, end, cm]]

    Returns:
        id_to_shared: dict of the form {i : L, ...}
            mapping each ID in id_set1 to the total
            amount shared with all people in id_set2
    """
    id_to_shared = {}
    for i in id_set1:
        shared_seg_list = get_segs_between_sets(
            id_set1={i},
            id_set2=id_set2,
            ibd_seg_list=ibd_seg_list,
        )
        merged_regions = get_merged_regions(shared_seg_list)
        L_tot = sum([e-s for c,s,e in merged_regions])
        id_to_shared[i] = L_tot
    return id_to_shared


def get_merged_regions(
    ibd_seg_list: list[list[int]],
):
    """
    Get all merged regions

    Args:
        ibd_seg_list: List of segments of the form
                        [[id1, id2, hap1, hap2, chrom, start, end, cm]]

    Returns:
        merged_regions: List of the form
                        [[chrom, start, end],...]
                        comprised of merged regions
    """
    if ibd_seg_list == []:
        return []

    ibd_seg_list = sorted(ibd_seg_list, key=lambda x: (x[4], x[5], x[6]))
    start_seg = ibd_seg_list[0]
    prev_chrom, prev_start, prev_end = start_seg[4:7]
    merged_regions = []
    for seg in ibd_seg_list:
        chrom, start, end = seg[4:7]
        if chrom == prev_chrom and start <= prev_end:
            prev_end = end
        else:
            write_seg = [prev_chrom, prev_start, prev_end]
            prev_chrom = chrom
            prev_start = start
            prev_end = end
            merged_regions.append(write_seg)
    write_seg = [prev_chrom, prev_start, prev_end]
    merged_regions.append(write_seg)
    return merged_regions


def get_total_ibd_between_id_sets(
    id_set1: set[int],
    id_set2: set[int],
    ibd_seg_list: list[list[int]],
):
    """
    Get the total amount of IBD shared between two sets of IDs

    Args:
        id_set1: set of genotyped IDs
        id_set2: set of genotyped IDs
        ibd_seg_list: List of segments of the form
                        [[id1, id2, hap1, hap2, chrom, start, end, cm]]

    Returns:
        L_tot: total amount of shared IBD

    FIXFIX: [EDIT:] this has been fixed in the case in which
            there is just one ID in id_set1 or id_set2. It is
            much harder to fix this for the general case because
            IBD need not be phased properly.

            This merges all segments assuming they
            are on the same haplotype. We actually
            need to account for phased IBD and count
            up the total IBD shared across both haplotypes.
            See the note on get_total_ibd_between_id_sets().

            Specifically, this currently assumes the following
            overlapping segments are really on the same haplotype
            and the total length is 16:

                0           12
                |           |
                ------------
                        --------
                        |       |
                        8       16

            merges to

                ----------------
                |               |
                0               16

            If these segments are actually on different haplotypes
            then we should not merge them. We should add them.
            The total length is the sum of their individual lengths,
            which is 20.

    """
    seg_subset = get_segs_between_sets(
        id_set1=id_set1,
        id_set2=id_set2,
        ibd_seg_list=ibd_seg_list,
    )

    # if id_set1 or id_set2 contains just one ID, then we should
    # merge the segments on the left and right haplotypes separately
    if len(id_set1) == 1:
        id1 = [*id_set1][0]
        seg_subset = [s for s in seg_subset if id1 in s[:2]]
        seg_subset0 = [s for s in seg_subset if s[2 + 1*(s[1]==id1)] == 0]
        seg_subset1 = [s for s in seg_subset if s[2 + 1*(s[1]==id1)] == 1]
        merged_regions0 = get_merged_regions(seg_subset0)
        merged_regions1 = get_merged_regions(seg_subset1)
        L_tot0 = sum([e-s for c,s,e in merged_regions0])
        L_tot1 = sum([e-s for c,s,e in merged_regions1])
        L_tot = L_tot0 + L_tot1
    elif len(id_set2) == 1:
        id2 = [*id_set2][0]
        seg_subset = [s for s in seg_subset if id2 in s[:2]]
        seg_subset0 = [s for s in seg_subset if s[2 + 1*(s[1]==id2)] == 0]
        seg_subset1 = [s for s in seg_subset if s[2 + 1*(s[1]==id2)] == 1]
        merged_regions0 = get_merged_regions(seg_subset0)
        merged_regions1 = get_merged_regions(seg_subset1)
        L_tot0 = sum([e-s for c,s,e in merged_regions0])
        L_tot1 = sum([e-s for c,s,e in merged_regions1])
        L_tot = L_tot0 + L_tot1
    else:
        merged_regions = get_merged_regions(seg_subset)
        L_tot = sum([e-s for c,s,e in merged_regions])

    return L_tot


def get_ibd_stats_frozenform(
    ibd_seg_list: list[list[int]],
):
    """
    Get the total number of segments and the total
    length of segments shared between each pair.

    Args:
        ibd_seg_list: List of segments of the form
                        [[id1, id2, hap1, hap2, chrom, start, end, cm]]

    Returns:
        ibd_stats: Dict of the form
            {
                frozenset({id1, id2}) : [num_segs, total_IBD],
                ...
            }
    """
    ibd_stats = {}
    for id1, id2, hap1, hap2, chrom_str, gen_start, gen_end, seg_len_cm in ibd_seg_list:

        key = frozenset({id1, id2})

        if key not in ibd_stats:
            ibd_stats[key] = [0, 0]

        ibd_stats[key][0] += 1
        ibd_stats[key][1] += seg_len_cm

    return ibd_stats


def get_ibd_stats_unphased(
    unphased_ibd_segs: list[tuple[int, int, str, float, float, bool, float]],
):
    """
    Get ibd_stats from ibd segments data

    Args:
        ibd_segs:
            [[id1, id2, chromosome, start, end, is_full_ibd, seg_cm]]

    Returns:
        ibd_stats:
            {frozenset({id1,id2}): {'total_half': in cm,
                                    'total_full': in cm,
                                    'num_half': int(total # of half segments),
                                    'max_seg_cm': largest half seg in cm}}
    """
    ibd_stats = defaultdict(lambda: {
        'total_half': 0,
        'total_full': 0,
        'num_half': 0,
        'num_full': 0,
        'max_seg_cm': 0})

    for s in unphased_ibd_segs:
        id1, id2, chromosome, start, end, is_full_ibd, seg_cm = s
        key = frozenset([id1, id2])
        ibd_stats[key]['total_half'] += (seg_cm if not is_full_ibd else 0)
        ibd_stats[key]['total_full'] += (seg_cm if is_full_ibd else 0)
        ibd_stats[key]['num_half'] += int(not is_full_ibd)
        ibd_stats[key]['num_full'] += int(is_full_ibd)
        ibd_stats[key]['max_seg_cm'] = max(ibd_stats[key]['max_seg_cm'], seg_cm)

    return ibd_stats


def get_closest_pair(
    ibd_stats: dict[frozenset, list[int]],
):
    """
    Find the pair of IDs that share
    the most total IBD.

    Args:
        ibd_stats: Dict of the form
            {
                frozenset({id1, id2}) : [num_segs, total_IBD],
                ...
            }

    Returns:
        i1,i2: IDs sharing the most total IBD
    """
    sorted_ibd_stats = sorted(
        ibd_stats.items(),
        key=lambda x: x[-1][-1],
        reverse=True,
    )

    pair, stat = sorted_ibd_stats[0]

    return [*pair]


def get_pair_to_chrom_to_ibd_segs(
    phased_ibd_seg_list: list[list[int]],
):
    """
    Get a dict mapping each pair of IDs to a dict
    mapping each chromosome to the list of segments
    shared between those IDs on that chromosome.

    Args:
        phased_ibd_seg_list: List of segments of the form
            [[id1, id2, hap1, hap2, chrom_str, start, end, cm]]

    Returns:
        pair_to_chrom_to_ibd_segs: Dict of the form
            {frozenset({i1,i2}) : chrom : (h1,h2) : [[start, end], ...], ...}
    """
    pair_to_chrom_to_ibd_segs = {}
    for i1, i2, h1, h2, chrom, start, end, cm in phased_ibd_seg_list:
        key = frozenset({i1, i2})
        if key not in pair_to_chrom_to_ibd_segs:
            pair_to_chrom_to_ibd_segs[key] = {}
        if chrom not in pair_to_chrom_to_ibd_segs[key]:
            pair_to_chrom_to_ibd_segs[key][chrom] = {}
        if (h1, h2) not in pair_to_chrom_to_ibd_segs[key][chrom]:
            pair_to_chrom_to_ibd_segs[key][chrom][(h1, h2)] = []
        pair_to_chrom_to_ibd_segs[key][chrom][(h1, h2)].append([start, end])
    return pair_to_chrom_to_ibd_segs


def get_pair_to_ibd_seg_len_list(
    phased_ibd_seg_list: list[list[int]],
):
    """
    Get a dict mapping each pair of IDs to the list
    of segment lengths shared between those IDs.

    Args:
        phased_ibd_seg_list: List of segments of the form
            [[id1, id2, hap1, hap2, chrom_str, start, end, cm]]

    Returns:
        pair_to_seg_len_list: Dict of the form
            {frozenset({i1,i2}) : [seg_len, ...], ...}
    """
    pair_to_seg_len_list = {}
    for i1, i2, h1, h2, chrom, start, end, cm in phased_ibd_seg_list:
        key = frozenset({i1, i2})
        if key not in pair_to_seg_len_list:
            pair_to_seg_len_list[key] = []
        seg_len = end - start
        pair_to_seg_len_list[key].append(seg_len)
    return pair_to_seg_len_list


def get_closest_pair_among_sets(
    id_set1: set[int],
    id_set2: set[int],
    pw_ll_cls: Any,
):
    """
    Find the ID in id_set1 and the ID in
    id_set2 that share the most total IBD.

    Args:
        id_set1: one set of IDs
        id_set2: another set of IDs
        pw_ll_cls: instance of class with methods for getting
                   pairwise log likelihoods.

    Returns:
        max_i1: ID in id_set1 that shares the most with
            any ID in id_set2
        max_i2: ID in id_set2 that shares the most with
            any ID in id_set1
        total_len: total IBD shared between i1 and i2
    """
    max_ibd = 0
    max_i1 = None
    max_i2 = None
    for i1 in id_set1:
        for i2 in id_set2:
            # chrom_to_hap_to_seg_list = pw_ll_cls.pair_to_chrom_to_ibd_segs[frozenset({i1, i2})]
            total_half = pw_ll_cls.ibd_stat_dict.get(frozenset({i1, i2}), {}).get('total_half', 0)
            total_full = pw_ll_cls.ibd_stat_dict.get(frozenset({i1, i2}), {}).get('total_full', 0)
            total = total_half + total_full
            if total > max_ibd:
                max_ibd = total
                max_i1 = i1
                max_i2 = i2
    return max_i1, max_i2, max_ibd


def get_id_set_from_seg_list(
    seg_list: list[list[int]],
):
    """
    Get all genotype IDs from a list of segments.

    Args:
        seg_list: Phased or unphased seg list of the form [[id1, id2, ...],...]

    Returns:
        id_set: Set of all genotype IDs in the seg list
    """
    id_set = set()
    for seg in seg_list:
        id_set.add(seg[0])
        id_set.add(seg[1])
    return id_set

from .constants import TWIN_THRESHOLD


def is_twin_pair(
    total_half: float,
    total_full: float,
    age1: int,
    age2: int,
    sex1: str,
    sex2: str,
):
    """
    Determine if a pair of individuals are twins.

    Args:
        total_half: The total length of half segments shared by the two individuals.
        total_full: The total length of full segments shared by the two individuals.
        age1: The age of the first individual.
        age2: The age of the second individual.
        sex1: Sex of the first individual.
        sex2: Sex of the second individual.

    Returns:
        True if the individuals are twins, False otherwise.
    """

    # handle unrelated people
    if total_half is None:
        return False
    if total_full is None:
        return False

    if total_half < TWIN_THRESHOLD:
        return False
    elif total_full < TWIN_THRESHOLD:
        return False
    elif sex1 != sex2:
        return False
    elif age1 and age2 and age1 != age2:
        return False

    return True


def get_twin_sets(
    ibd_stat_dict: dict[frozenset, dict[str, int]],
    age_dict: dict[int, float],
    sex_dict: dict[int, str],
):
    """
    Find all sets of twins.

    Args:
        ibd_stat_dict: A dictionary of IBD statistics.
        age_dict: A dictionary mapping ID to age
        sex_dict: A dictionary mapping ID to sex ('M' or 'F')

    Returns:
        idx_to_twin_set: A dictionary mapping an index to a set of node IDs
            that form a twin set.
        id_to_idx: A dict mapping each twin ID to its index.
    """
    idx_to_twin_set = {}
    id_to_idx = {}
    ctr = 0
    for k,v in ibd_stat_dict.items():
        id1, id2 = [*k]
        age1 = age_dict.get(id1)
        age2 = age_dict.get(id2)
        sex1 = sex_dict.get(id1)
        sex2 = sex_dict.get(id2)

        total_half = v.get("total_half")
        total_full = v.get("total_full")

        is_twin = is_twin_pair(
            total_half = total_half,
            total_full = total_full,
            age1 = age1,
            age2 = age2,
            sex1 = sex1,
            sex2 = sex2,
        )

        if is_twin:
            idx1 = id_to_idx.get(id1, ctr)
            idx2 = id_to_idx.get(id2, ctr)

            # Add the IDs
            idx_to_twin_set.setdefault(idx1, set()).add(id1)
            idx_to_twin_set.setdefault(idx2, set()).add(id2)

            # combine sets
            if idx1 != idx2:
                idx_to_twin_set[idx1] |= idx_to_twin_set[idx2]

            for iid in idx_to_twin_set[idx1]:
                id_to_idx[iid] = idx1

            # delete idx2 if it is not ctr
            if idx1 != idx2:
                del idx_to_twin_set[idx2]

            ctr += 1

    return idx_to_twin_set, id_to_idx

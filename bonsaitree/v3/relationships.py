from typing import Optional

from .constants import INF


def reverse_rel(
    rel : Optional[tuple[int, int, int]],
):
    """
    Reverse a relationship tuple (u, d, a)
    to (d, u, a) or None to None.

    Args:
        rel: relationship tuple of the form (u, d, a) or None

    Returns:
        rev_rel: (d, u, a) or None
    """
    if type(rel) is tuple:
        rev_rel = (rel[1], rel[0], rel[2])
    else:
        rev_rel = rel
    return rev_rel


def get_deg(
    rel : Optional[tuple[int, int, int]],
):
    """
    Get the degree of a relationship tuple.

    Args:
        rel: relationship tuple of the form (u, d, a), (m, a), or None

    Returns:
        deg: if rel is None or a is None: INF
             else: u + d - a + 1
    """
    if type(rel) is tuple:

        # get m and a
        if len(rel) == 3:
            u,d,a = rel
            m = u+d
        elif len(rel) == 2:
            m, a = rel

        # get degree
        if a is None:
            deg = INF
        elif m == 0:
            deg = 0
        else:
            deg = m - a + 1
    else:
        deg = INF
    return deg


def join_rels(
    rel_ab : Optional[tuple[int, int, int]],
    rel_bc : Optional[tuple[int, int, int]],
    # num_ancs : int,
):
    """
    For three individuals A, B, and C
    related by relAB and relBC, find relAC.

    Args:
        rel_ab: relationship between A and B of the form (u, d, a) or None
        rel_bc: relationship between B and C of the form (u, d, a) or None
        # num_ancs: number of ancestors connecting A and C in the joined path
        #           relAB + relBC

    Returns:
        relAC: relationship between A and C of the form (u, d, a) or None
    """

    if (rel_ab is None) or (rel_bc is None):
        return None

    if rel_ab == (0, 0, 2):
        return rel_bc

    if rel_bc == (0, 0, 2):
        return rel_ab

    u1, d1, a1 = rel_ab
    u2, d2, a2 = rel_bc

    if a1 is None or a2 is None:
        return None

    if d1 > 0 and u2 > 0:
        return None

    if u1 > 0 and d1 > 0 and u2 == 0 and d2 > 0:
        a = a1
    elif u1 == 0 and d1 > 0 and u2 == 0 and d2 > 0:
        a = a2
    elif u1 > 0 and d1 == 0 and u2 > 0 and d2 == 0:
        a = a2
    elif u1 > 0 and d1 == 0 and u2 > 0 and d2 > 0:
        a = a2
    elif u1 > 0 and d1 == 0 and u2 == 0 and d2 > 0:
        a = a1
    elif u1 > 0 and d1 > 0 and u2 == 0 and d2 == 0:
        a = a1
    elif u1 == 0 and d1 > 0 and u2 == 0 and d2 == 0:
        a = a1
    elif u1 > 0 and d1 == 0 and u2 == 0 and d2 == 0:
        a = a1
    elif u1 == 0 and d1 == 0 and u2 == 0 and d2 > 0:
        a = a2
    elif u1 == 0 and d1 == 0 and u2 > 0 and d2 > 0:
        a = a2

    u = u1 + u2
    d = d1 + d2

    return (u, d, a)


def get_transitive_rel(
    rel_list : list[Optional[tuple[int, int, int]]],
    # num_ancs : List[int],
):
    """
    For a list of relationships in rel_list that represent
    pairwise relationships from one person to the next in a chain
    of relatives, find the relationship between the first person
    in the list and the last person in the list.

    Args:
        rel_list: chain of relationships of the form [(up, down, num_ancs), ...]
                  where rel_list[i] is the relationship between individuals i and i+1.
        # num_ancs : int. Number of common ancestors shared between person 1
        #            and person 2 if that number is otherwise ambiguous from the
        #            list of relationship tuples.

    Returns:
        rel: The relationship between individual 0 and individual n specified
             by the chain of relationships in rel_list.
    """
    if rel_list == []:
        return None

    rel = rel_list.pop(0)
    while rel_list:
        next_rel = rel_list.pop(0)
        rel = join_rels(rel, next_rel)  #, num_ancs)

    # ensure that ancestor/descendant relationships
    # only have one ancestor. We can't ensure this
    # within join_rels() so we have to do it here.
    if rel is not None and (rel[0] == 0 or rel[1] == 0):
        rel = (rel[0], rel[1], 1)

    return rel


def a_m_to_rel(
    a : int,
    m : int,
):
    """
    Convert number of meioses and number of common ancestors
    to a good approximate relationship of the form (up, down, num_common_ancs).

    Note that m and a do not capture the full distribution so
    this is a one-to-many map and we choose the "best" relationship,
    which is a collateral relationship, which I assume will be more
    common in the pedigrees we encounter.

    The distribution is "similar" for direct ancestral relationships
    so I think we can use it.

    Args:
        a: number of common ancestors
        m: number of meioses

    Returns:
        rel: "best" relationship corresponding to a and m
    """

    if m == 1:
        rel = (m, 0, 1)
    elif m == 0 and a == 2:
        rel = (0, 0, 2)
    else:
        rel = (m-1, 1, a)

    return rel

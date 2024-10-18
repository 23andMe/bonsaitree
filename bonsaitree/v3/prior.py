from typing import Union

import numpy as np
import numpy.typing as npt

from .constants import C, Q, R
from .likelihoods import get_eta


# @functools.cache
def exp_num_female_common_ancs(
    N : float,
    g : Union[int, npt.ArrayLike],
):
    """
    Calculate the expected number of distinct common female ancestors
    at generation g in the past for a population of size N.

    Args:
        N: float, population size
        g: int or array of ints, number of generations in the past

    Returns:
        expected number of distinct female common ancestors.
    """
    return N * (1 - (1 - 1/N)**(2**(g-1)))


# @functools.cache
def exp_num_total_common_ancs(
    N : float,
    g : Union[int, npt.ArrayLike],
):
    """
    Calculate the expected total number of distinct common ancestors
    at generation g in the past for a population of size N.

    Args:
        N: float, population size
        g: int or array of ints, number of generations in the past

    Returns:
        expected number of distinct common ancestors.
    """
    return 2 * exp_num_female_common_ancs(N, g)


# @functools.cache
def exp_num_lins_per_anc(
    N : float,
    g : Union[int, npt.ArrayLike],
):
    """
    Calculate the expected number of lineages back to each ancestor
    at generation g in the past for a population of size N.

    Args:
        N: float, population size
        g: int or array of ints, number of generations in the past

    Returns:
        expected number of lineages back to each ancestor.
    """
    num_lins = 2**(g-1)
    exp_num_ancs = exp_num_female_common_ancs(N, g)
    return num_lins / exp_num_ancs


# @functools.cache
def exp_num_shared_common_ancs(
    N : float,
    g : Union[int, npt.ArrayLike],
):
    """
    Get the expected number of shared common ancestors
    between two people at g generations in the past.

    Args:
        N: float, population size
        g: int or array of ints, number of generations in the past

    Returns:
        expected number of shared common ancestors.
    """
    return 2 * N * ((1 - (1 - 1/N)**(2**(g-1)))**2)


# @functools.cache
def prob_share_any_ibd_seg(
    m_lst : Union[int, float, npt.ArrayLike],
    a : int,
    min_seg_len : float,
    r=R,
    c=C,
    q=Q,
):
    """
    Calculate the probability that two individuals share any IBD segment
    given that they are separated by m meioses with a common ancestor
    for m in m_lst and a in a_lst.

    Args:
        m_lst: int or array of ints, number(s) of meioses separating the two individuals
        a: int, number of common ancestors
        min_seg_len: float, minimum observable segment length
        r: float, expected number of recombinations per genome per generation
        c: int, number of autosomes
        q: float, parameter of the false negative rate for low coverage segments

    Returns:
        probability that two individuals share any IBD segment.
    """
    eta = get_eta(
        a_lst=a,
        m_lst=m_lst,
        min_seg_len=min_seg_len,
        r=r,
        c=c,
        q=q,
    )
    return 1 - np.exp(-eta)


# @functools.cache
def prob_share_no_ibd_seg(
    m_lst : Union[int, float, npt.ArrayLike],
    a : int,
    min_seg_len : float,
    r=R,
    c=C,
    q=Q,
):
    """
    Calculate the probability that two individuals share no IBD segment
    given that they are separated by d meioses with a common ancestors.

    Args:
        m_lst: int or array of ints, number(s) of meioses separating the two individuals
        a: int, number of common ancestors
        min_seg_len: float, minimum observable segment length
        r: float, expected number of recombinations per genome per generation
        c: int, number of autosomes
        q: float, parameter of the false negative rate for low coverage segments

    Returns:
        probability that two individuals share no IBD segment.
    """
    prob_any_ibd = prob_share_any_ibd_seg(
        m_lst=m_lst,
        a=a,
        min_seg_len=min_seg_len,
        r=r,
        c=c,
        q=q,
    )
    return 1 - prob_any_ibd


# @functools.cache
def get_exp_num_ibd_trans_ancs(
    g_range : npt.ArrayLike,
    N : float,
    a : int,
    min_seg_len : float,
    r : float=R,
    c : int=C,
    q : float=Q,
):
    """
    Get the prior probability that an IBD-transmitting common ancestor
    between two people in the same generation lived g generations in the past.

    Args:
        g_range: list of integers, generations in the past to consider
        N: int, population size
        a: int, number of common ancestors
        min_seg_len: float, minimum observable segment length
        r: float, expected number of recombinations per genome per generation
        c: int, number of autosomes
        q: float, parameter of the false negative rate for low coverage segments

    Returns:
        pmf: list of floats, prior probability mass function of the
            number of generations in the past when the IBD-transmitting
            common ancestor lived.
    """
    pmf = []

    # get the expected number of common ancestors at each generation in g_range
    exp_num_common_ancs = exp_num_shared_common_ancs(N=N, g=g_range)

    # get the probability that an ancestor in generation g transmits detectable IBD
    m_lst = 2 * g_range
    prob_share_ibd = prob_share_any_ibd_seg(
        m_lst=m_lst,
        a=a,
        min_seg_len=min_seg_len,
        r=r,
        c=c,
        q=q,
    )

    # get the expected number of IBD-transmitting common ancestors in each generation
    exp_num_trans_common_ancs = exp_num_common_ancs * prob_share_ibd

    return exp_num_trans_common_ancs[0]


# @functools.cache
def get_prior_g(
    g_range : npt.ArrayLike,
    N : float,
    a : int,
    min_seg_len : float,
    r : float=R,
    c : int=C,
    q : float=Q,
):
    """
    Get the prior probability that an IBD-transmitting common ancestor
    between two people in the same generation lived g generations in the past.

    Args:
        g_range: list of integers, generations in the past to consider
        N: int, population size
        a: int, number of common ancestors
        min_seg_len: float, minimum observable segment length
        r: float, expected number of recombinations per genome per generation
        c: int, number of autosomes
        q: float, parameter of the false negative rate for low coverage segments

    Returns:
        pmf: list of floats, prior probability mass function of the
            number of generations in the past when the IBD-transmitting
            common ancestor lived.
    """
    exp_num_trans_common_ancs = get_exp_num_ibd_trans_ancs(
        g_range=g_range,
        N=N,
        a=a,
        min_seg_len=min_seg_len,
        r=r,
        c=c,
        q=q,
    )

    # normalize the pmf
    pmf = exp_num_trans_common_ancs / exp_num_trans_common_ancs.sum()

    return pmf

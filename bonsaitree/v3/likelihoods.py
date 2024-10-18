from itertools import combinations
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pyximport
import scipy.stats
from scipy.special import gammaln, iv, logsumexp

pyximport.install()
from .constants import (  # noqa: E402
    DEG_CI_ALPHA,
    GENOME_LENGTH,
    INF,
    MEAN_BGD_LEN,
    MEAN_BGD_NUM,
    MIN_MEAN_BGD_LEN,
    MIN_MEAN_BGD_NUM,
    MIN_SEG_LEN,
    PW_TOTAL_IBD_M_LIST,
    UNREL_DEG,
    C,
    P,
    Q,
    R,
)
from .cythonized.likelihood import get_L_log_pdf_approx, get_n_L_log_pdf_approx  # noqa: E402
from .exceptions import BonsaiError  # noqa: E402
from .ibd import (  # noqa: E402
    get_ibd_stats_unphased,
)
from .imports import load_ibd_moments  # noqa: E402
from .moments import (  # noqa: E402
    load_age_moments,
)
from .pedigrees import (  # noqa: E402
    get_gt_id_set,
    get_simple_rel_tuple,
)
from .relationships import (  # noqa: E402
    get_deg,
    reverse_rel,
)

MAX_DEG = -INF


def get_lam_a_m(
    m_lst : npt.ArrayLike,
    covs1 : Optional[npt.ArrayLike]=None,
    covs2 : Optional[npt.ArrayLike]=None,
    p : Optional[float]=P,
):
    """
    Get the inverse mean segment length for a relationship
    with m meioses and a common anceestors.

    Args:
        m_lst: array-like object of possible numbers of meioses
        covs1: array-like: element i is coverage of person i
        covs2: array-like: element i is coverage of person i
        p: low coverage segment length scaling factor (found empirically)
           See: https://www.science.org/doi/10.1126/
                Equation S6.4

    Returns:
        lam_a_m: lam_a_m[i][j] is the inverse mean segment length
                 evaluated at meiosis m_lst[j] for a pair of people
                 with coverages cov1 = covs1[i] and cov2 = covs2[i].
    """

    if covs1 is None:
        covs1 = np.array([INF])
    if covs2 is None:
        covs2 = np.array([INF])

    lam_a_m = m_lst/100

    # adjust mean segment lengths for close relationships
    lam_a_m[m_lst == 1] = 1/60  # approximately the empirical mean
    lam_a_m[m_lst == 2] = 1/38  # approximately the empirical mean
    lam_a_m[m_lst == 3] = 1/28  # approximately the empirical mean
    lam_a_m[m_lst == 4] = 1/22  # approximately the empirical mean
    lam_a_m[m_lst == 5] = 1/18  # approximately the empirical mean
    lam_a_m[m_lst == 6] = 1/15  # approximately the empirical mean

    cov_factor1 = 1 - np.exp(-p*covs1)
    cov_factor2 = 1 - np.exp(-p*covs2)
    cov_factor = cov_factor1 * cov_factor2

    lam_a_m = np.outer(cov_factor, lam_a_m)

    return lam_a_m


def get_eta(
    a_lst : npt.ArrayLike,
    m_lst : npt.ArrayLike,
    covs1 : Optional[npt.ArrayLike]=None,
    covs2 : Optional[npt.ArrayLike]=None,
    min_seg_len : float=0,
    r : float=R,
    c : int=C,
    q : float=Q,
):
    """
    Get the expected number of segments for a relationship with
    m meioses, a common ancestors, and a minimum segment length
    of min_seg_len.

    Args:
        a_lst: array-like object of possible numbers of common ancestors
        m_lst: array-like object of possible numbers of meioses. Same length
            as a_lst
        covs1: array-like: element i is coverage of person i
        covs2: array-like: element i is coverage of person i
        min_seg_len: minimum observable segment length
        r: expected number of recombination events per genome per meiosis
        c: number of autosomes
        q: false negative parameter for low coverage genomes
           See: https://www.science.org/doi/10.1126/
                Equation S6.3

    Returns:
        lam_a_m_t: lam_a_m_t[i][j] is the expected number of segments
                   at meiosis m_lst[j] for a pair with coverages cov1 = covs1[i]
                   and cov2 = covs2[i].
    """

    if covs1 is None:
        covs1 = np.array([INF])
    if covs2 is None:
        covs2 = np.array([INF])

    # construct an array of 1s for outer products
    num_cols = len(m_lst)
    num_rows = len(covs1)
    col_zeros = np.zeros(num_cols)
    row_zeros = np.zeros(num_rows)

    # m needs to be a float for negative powers
    m_lst = m_lst.astype(float)

    # get lambda (expected number of segments), unadjusted for coverage
    eta = (2**(1-m_lst)) * a_lst * (r*m_lst+c)

    # expand lam_amt, m_lst, covs1, and covs2 into 2D arrays
    eta_arr = np.add.outer(row_zeros, eta)
    m_arr = np.add.outer(row_zeros, m_lst)
    cov1_arr = np.add.outer(covs1, col_zeros)
    cov2_arr = np.add.outer(covs2, col_zeros)

    # correct for coverage
    if min_seg_len > 0:
        eta_arr *= (
            np.exp(-min_seg_len*m_arr/100) -
            (
                (m_arr/100) *
                np.exp(-(m_arr/100 + q*cov1_arr)*min_seg_len) /
                (m_arr/100 + q*cov1_arr)
            ) -
            (
                (m_arr/100) *
                np.exp(-(m_arr/100 + q*cov2_arr)*min_seg_len) /
                (m_arr/100 + q*cov2_arr)
            ) +
            (
                (m_arr/100) *
                np.exp(-(m_arr/100 + q*(cov1_arr+cov2_arr))*min_seg_len) /
                (m_arr/100 + q*(cov1_arr+cov2_arr))
            )
        )
    else:
        eta_arr *= (
            np.exp(-min_seg_len*m_arr/100) -
            (m_arr/100) / (m_arr/100 + q*cov1_arr) -
            (m_arr/100) / (m_arr/100 + q*cov2_arr) +
            (m_arr/100) / (m_arr/100 + q*(cov1_arr+cov2_arr))
        )

    return eta_arr


def get_poisson_log_pmf_arr(
    ks : npt.ArrayLike,
    lambdas : npt.ArrayLike,
):
    """
    Get the log Poisson pmf evaluated at each
    n,lambda in the 2D arrays ks and lambdas

    Args:
        ks: 2D numpy array of counts
        lambdas: 2D numpy array of Poisson lambdas

    Returks:
        log_pmf: element i,j is the PMF of the
                 Poisson distribution evaluated at ks[i][j]
                 with parameter lambdas[i][j].
    """

    log_pmf = ks * np.log(lambdas) - lambdas - gammaln(ks+1)

    return log_pmf


def get_seg_len_log_pdf_unconditional(
    seg_len_list: npt.ArrayLike,
    m_arr: npt.ArrayLike,
    a: int,
    mean_len_bgd: float=MEAN_BGD_LEN,
    mean_num_bgd: float=MEAN_BGD_NUM,
    min_seg_len: float=MIN_SEG_LEN,
    covs1: Optional[npt.ArrayLike]=None,
    covs2: Optional[npt.ArrayLike]=None,
    r: float=R,
    c: int=C,
    q: float=Q,
    min_mean_num_bgd: float=MIN_MEAN_BGD_NUM,
    min_mean_len_bgd: float=MIN_MEAN_BGD_LEN,
):
    """
    Do not condition on observing at least one segment.

    Compute the log pdf of the set of segment lengths
    when between two people separated by m meioses and
    sharing a common ancestors when at least one segment
    is observed between them.

    Args:
        seg_len_list: list of segment lengths
        m_arr: array containing numbers of meioses separating the people
        a: number of common ancestors
        mean_len_bgd: mean length of background segments
        mean_num_bgd: mean number of background segments
        min_seg_len: minimum observable segment length
        covs1: array-like: repeated coverage of person 1
        covs2: array-like: repeated coverage of person 2
        r: expected number of recombination events per genome per meiosis
        c: number of autosomes
        q: false negative parameter for low coverage genomes
              See: https://www.science.org/doi/10.1126/
        min_mean_num_bgd: minimum expected number of background segments
        min_mean_len_bgd: minimum expected length of background segments

    Returns:
        log_pdf: log pdf of the segment lengths evaluated
                 at each value of m_arr.
    """

    if (mean_num_bgd == 0) and (mean_len_bgd > 0):
        raise BonsaiError("mean_len_bgd must be 0 if mean_num_bgd is 0.")

    if covs1 is None:
        covs1 = INF * np.ones(len(m_arr))
    if covs2 is None:
        covs2 = INF * np.ones(len(m_arr))

    # get expected number of foreground segments
    if a is None:
        eta = max(mean_num_bgd, min_mean_num_bgd)
        eta = np.array([[eta]])
    else:
        eta = get_eta(
            a_lst=np.array([a]),
            m_lst=m_arr,
            min_seg_len=min_seg_len,
            covs1=covs1,
            covs2=covs2,
            r=r,
            c=c,
            q=q,
        )

    if a is None:
        lam_am = (np.zeros(len(m_arr))+1) / max(mean_len_bgd, min_mean_len_bgd)
        lam_am = np.array([lam_am])
    else:
        m_lst = m_arr
        lam_am = get_lam_a_m(
            m_lst=m_lst,
            covs1=covs1,
            covs2=covs2,
        )

    # get the number of segments
    num_segs = len(seg_len_list)

    # handle the case in which there are no observed segments
    # in this case, the probability is just the probaiblity that
    # there are no segments at all.
    if num_segs == 0:
        log_pdf = -eta
        return log_pdf

    # if the expected number of background segments is 0
    if mean_num_bgd == 0:
        log_pdf = (
            num_segs * np.log(lam_am)
            - lam_am * sum(seg_len_list)
            + lam_am * num_segs * min_seg_len
            + num_segs * np.log(eta)
            - eta
            - gammaln(num_segs+1)
        )

        return log_pdf

    # get the list of terms in the summation (Eqn 8 of segment-based likelihoods as of 2024/03/25)
    term_list = []
    for i in range(0, num_segs+1):
        term = (
            i * np.log(lam_am)
            - (lam_am) * sum(seg_len_list[:i])
            + (lam_am) * i * min_seg_len
            - (num_segs-i) * np.log(mean_len_bgd)
            - (1/mean_len_bgd) * sum(seg_len_list[i:])
            + (num_segs-i) * min_seg_len / mean_len_bgd
            + (num_segs-i) * np.log(mean_num_bgd)
            - mean_num_bgd
            + i * np.log(eta)
            - eta
            - gammaln(i+1)
            - gammaln(num_segs-i+1)
        )

        term_list.append(term)

    log_pdf = logsumexp(term_list, axis=0)

    return log_pdf


def get_seg_len_log_pdf_conditional(
    seg_len_list: npt.ArrayLike,
    m_arr: npt.ArrayLike,
    a: int,
    mean_len_bgd: float=MEAN_BGD_LEN,
    mean_num_bgd: float=MEAN_BGD_NUM,
    min_seg_len: float=MIN_SEG_LEN,
    covs1: Optional[npt.ArrayLike]=None,
    covs2: Optional[npt.ArrayLike]=None,
    r: float=R,
    c: int=C,
    q: float=Q,
    min_mean_num_bgd: float=MIN_MEAN_BGD_NUM,
    min_mean_len_bgd: float=MIN_MEAN_BGD_LEN,
):
    """
    Condition on observing at least one segment.

    Compute the log pdf of the set of segment lengths
    when between two people separated by m meioses and
    sharing a common ancestors when at least one segment
    is observed between them.

    Args:
        seg_len_list: list of segment lengths
        m_arr: array containing numbers of meioses separating the people
        a: number of common ancestors
        mean_len_bgd: mean length of background segments
        mean_num_bgd: mean number of background segments
        min_seg_len: minimum observable segment length
        covs1: array-like: repeated coverage of person 1
        covs2: array-like: repeated coverage of person 2
        r: expected number of recombination events per genome per meiosis
        c: number of autosomes
        q: false negative parameter for low coverage genomes
              See: https://www.science.org/doi/10.1126/
        min_mean_num_bgd: minimum expected number of background segments
        min_mean_len_bgd: minimum expected length of background segments

    Returns:
        log_pdf: log pdf of the segment lengths evaluated
                 at each value of m_arr.
    """

    if (mean_num_bgd == 0) and (mean_len_bgd > 0):
        raise BonsaiError("mean_len_bgd must be 0 if mean_num_bgd is 0.")

    if covs1 is None:
        covs1 = INF * np.ones(len(m_arr))
    if covs2 is None:
        covs2 = INF * np.ones(len(m_arr))

    # get expected number of foreground segments
    if a is None:
        eta = max(mean_num_bgd, min_mean_num_bgd)
        eta = np.array([[eta]])
    else:
        eta = get_eta(
            a_lst=np.array([a]),
            m_lst=m_arr,
            min_seg_len=min_seg_len,
            covs1=covs1,
            covs2=covs2,
            r=r,
            c=c,
            q=q,
        )

    if a is None:
        lam_am = (np.zeros(len(m_arr))+1) / max(mean_len_bgd, min_mean_len_bgd)
        lam_am = np.array([lam_am])
    else:
        m_lst = m_arr
        lam_am = get_lam_a_m(
            m_lst=m_lst,
            covs1=covs1,
            covs2=covs2,
        )

    # get the number of segments
    num_segs = len(seg_len_list)

    # handle the case in which there are no observed segments
    # in this case, the probability is just the probaiblity that
    # there are no segments at all. Since we are conditioning on
    # observing at least one segment, this probability is 0.
    if num_segs == 0:
        log_pdf = -INF
        return np.array([[log_pdf]])

    # if the expected number of background segments is 0
    if mean_num_bgd == 0:
        log_pdf = (
            num_segs * np.log(lam_am)
            - lam_am * sum(seg_len_list)
            + lam_am * num_segs * min_seg_len
            + num_segs * np.log(eta)
            - eta
            - gammaln(num_segs+1)
        )
        # condition on observing at least one segment
        norm_term = np.log(1 - np.exp(-eta))[0]
        inf_mask = (eta < 1e-3)[0]
        approx_term = np.log(eta)[0]
        norm_term[inf_mask] = approx_term[inf_mask]
        log_pdf -= norm_term

        return log_pdf

    # get the list of terms in the summation (Eqn 8 of segment-based likelihoods as of 2024/03/25)
    term_list = []
    for i in range(0, num_segs+1):
        term = (
            i * np.log(lam_am)
            - (lam_am) * sum(seg_len_list[:i])
            + (lam_am) * i * min_seg_len
            - (num_segs-i) * np.log(mean_len_bgd)
            - (1/mean_len_bgd) * sum(seg_len_list[i:])
            + (num_segs-i) * min_seg_len / mean_len_bgd
            + (num_segs-i) * np.log(mean_num_bgd)
            - mean_num_bgd
            + i * np.log(eta)
            - eta
            - gammaln(i+1)
            - gammaln(num_segs-i+1)
        )

        # condition on observing at least one segment
        norm_term = np.log(1 - np.exp(-mean_num_bgd - eta))[0]
        inf_mask = (mean_num_bgd+eta < 1e-3)[0]
        approx_term = np.log(mean_num_bgd+eta)[0]
        norm_term[inf_mask] = approx_term[inf_mask]
        term -= norm_term

        term_list.append(term)

    log_pdf = logsumexp(term_list, axis=0)

    return log_pdf


def total_conditional_ibd_pdf_bessel(
    L : float,
    m : int,
    a : int,
    cov1 : Optional[float]=None,
    cov2 : Optional[float]=None,
    min_seg_len : float=MIN_SEG_LEN,
    r : float=R,
    c : int=C,
    q : float=Q,
):

    covs1 = None if cov1 is None else np.array([cov1])
    covs2 = None if cov2 is None else np.array([cov2])

    lam = get_lam_a_m(
        m_lst=np.array([m]),
        covs1=covs1,
        covs2=covs2,
    )[0][0]

    eta = get_eta(
        a_lst = np.array([a]),
        m_lst = np.array([m]),
        covs1 = covs1,
        covs2 = covs2,
        min_seg_len = min_seg_len,
        r = r,
        c = c,
        q = q,
    )[0][0]

    pdf = (
        np.exp(-lam*(L-min_seg_len)) *
        np.sqrt(eta*lam) *
        iv(1, 2*np.sqrt((L-min_seg_len)*eta*lam)) /
        (np.sqrt(L-min_seg_len) *
        (np.exp(eta) - 1))
    )

    return pdf


def total_unconditional_ibd_pdf_bessel(
    L : float,
    m : int,
    a : int,
    cov1 : Optional[float]=None,
    cov2 : Optional[float]=None,
    min_seg_len : float=MIN_SEG_LEN,
    r : float=R,
    c : int=C,
    q : float=Q,
):

    covs1 = None if cov1 is None else np.array([cov1])
    covs2 = None if cov2 is None else np.array([cov2])

    lam = get_lam_a_m(
        m_lst=np.array([m]),
        covs1=covs1,
        covs2=covs2,
    )[0][0]

    eta = get_eta(
        a_lst = np.array([a]),
        m_lst = np.array([m]),
        covs1 = covs1,
        covs2 = covs2,
        min_seg_len = min_seg_len,
        r = r,
        c = c,
        q = q,
    )[0][0]

    pdf = (
        np.exp(-eta) *
        np.exp(-lam*(L-min_seg_len)) *
        np.sqrt(eta*lam) *
        iv(1, 2*np.sqrt((L-min_seg_len)*eta*lam)) /
        np.sqrt(L-min_seg_len)
    )

    return pdf


def total_conditional_ibd_and_seg_ct_log_pdf(
    L : float,
    n : int,
    m : int,
    a : int,
    cov1 : Optional[float]=None,
    cov2 : Optional[float]=None,
    min_seg_len : float=MIN_SEG_LEN,
    r : float=R,
    c : int=C,
    q : float=Q,
):

    covs1 = None if cov1 is None else np.array([cov1])
    covs2 = None if cov2 is None else np.array([cov2])

    lam = get_lam_a_m(
        m_lst=np.array([m]),
        covs1=covs1,
        covs2=covs2,
    )[0][0]

    eta = get_eta(
        a_lst = np.array([a]),
        m_lst = np.array([m]),
        covs1 = covs1,
        covs2 = covs2,
        min_seg_len = min_seg_len,
        r = r,
        c = c,
        q = q,
    )[0][0]

    log_pdf = (
        n*np.log(lam) +
        (n-1)*np.log(L) -
        lam*L +
        np.log(eta) -
        eta -
        np.log(1 - np.exp(-eta)) -
        gammaln(n) -
        gammaln(n+1)
    )

    return log_pdf


def total_unconditional_ibd_and_seg_ct_log_pdf(
    L : float,
    n : int,
    m : int,
    a : int,
    cov1 : Optional[float]=None,
    cov2 : Optional[float]=None,
    min_seg_len : float=MIN_SEG_LEN,
    r : float=R,
    c : int=C,
    q : float=Q,
):

    covs1 = None if cov1 is None else np.array([cov1])
    covs2 = None if cov2 is None else np.array([cov2])

    lam = get_lam_a_m(
        m_lst=np.array([m]),
        covs1=covs1,
        covs2=covs2,
    )[0][0]

    eta = get_eta(
        a_lst = np.array([a]),
        m_lst = np.array([m]),
        covs1 = covs1,
        covs2 = covs2,
        min_seg_len = min_seg_len,
        r = r,
        c = c,
        q = q,
    )[0][0]

    log_pdf = (
        n*np.log(lam) +
        (n-1)*np.log(L) -
        lam*L +
        np.log(eta) -
        eta -
        gammaln(n) -
        gammaln(n+1)
    )

    return log_pdf


def get_age_mean_std_for_rel_tuple(
    rel_tuple : Optional[tuple[int, int, int]],
    age1 : Optional[int]=None,
    age2 : Optional[int]=None,
):
    """
    Get the expected age difference and the
    standard deviation of that difference
    for a pair related through rel_tuple.

    Args:
        rel_tuple: None (if unrelated) or tuple of the
            form (up, down, num_ancs)

    Returns:
        mean: mean age difference (age2 - age1)
        std: standard deviation of age difference (age2 - age1)
    """

    # load memomized age moments
    age_moments = load_age_moments()

    if rel_tuple in age_moments:
        mean, std = age_moments[rel_tuple]
    else:
        # if we don't have the empirical moments, we'll
        # need to guess them by assuming that the mean
        # and variance are the same as for a string of
        # parent child relationships attaching the two
        # people.

        # get parent/child mean and standard deviation
        pc_tuple = (0, 1, 1)
        pc_mean, pc_std = age_moments[pc_tuple]
        pc_var = pc_std**2

        # get full sibling mean and var
        fs_tuple = (1, 1, 2)
        fs_mean, fs_std = age_moments[fs_tuple]
        fs_var = fs_std**2

        # get half sibling mean and var
        hs_tuple = (1, 1, 1)
        hs_mean, hs_std = age_moments[hs_tuple]
        hs_var = hs_std**2

        if rel_tuple is None:  # assume unrelateds have mean/std that are consitent with observation

            if (age1 is None) or (age2 is None):
                raise Exception("If rel_tuple is None, age1 and age2 must not be None.")

            mean = age1 - age2  # assume the mean is the observed mean
            var = 2 * pc_var  # 2X the variance of a P/C relationship
            std = np.sqrt(var)
        else:
            # the mean will be m times the parent-child mean where m is the total number
            # of meioses between node1 and node2. The variance will be m times the parent-child
            # variance if node1 is related to node2 through a single common ancestor and it will
            # be the full sibling variance, plus (m-2)*pc_var if they are related through two
            # common ancestors
            gen_diff = rel_tuple[1] - rel_tuple[0]
            gen_sum = rel_tuple[0] + rel_tuple[1]
            mean = gen_diff * pc_mean

            if rel_tuple[0] == 0 or rel_tuple[1] == 0:
                var = gen_sum * pc_var
            elif rel_tuple[2] == 1:
                var = (gen_sum-2) * pc_var + hs_var
            elif rel_tuple[2] == 2:
                var = (gen_sum - 2) * pc_var + fs_var

            std = np.sqrt(var)

    return mean, std


def get_age_log_like(
    age1 : int,
    age2 : int,
    rel_tuple : Optional[tuple[int, int, int]],
):
    """
    Find the age component of the pairwise
    log likelihood.

    Args:
        age1 : age of first person
        age2 : age of second person
        rel_tuple : None (if unrelated) or tuple
            of the form (up, down, num_ancs)

    Returns:
        pw_ll : age component of the pairwise
            log likelihood expressing the log
            likelihood that the two people are
            related through relationship rel_tuple.
    """
    # don't count ages towards the likelihood if they are unknown
    if age1 is None or age2 is None:
        return 0
    mean, std = get_age_mean_std_for_rel_tuple(
        rel_tuple=rel_tuple,
        age1=age1,
        age2=age2,
    )
    age_diff = age1 - age2
    return scipy.stats.norm.logpdf(age_diff, loc=mean, scale=std)


def get_age_logpdf(
    age1 : int,
    age2 : int,
    rel_tuple : Optional[tuple[int, int, int]],
):
    """
    Find the value of the PDF of the age distribution
    for ages 1 and 2 related through relationship rel_tuple.

    Args:
        age1 : age of first person
        age2 : age of second person
        rel_tuple : None (if unrelated) or tuple
            of the form (up, down, num_ancs)

    Returns:
        logpdf : age component of the pairwise
            log likelihood expressing the log
            likelihood that the two people are
            related through relationship rel_tuple.
    """
    # handle the case in which we don't have the age of one of the people.
    # In this case, an age likelihood involving that person should always
    # be of the same magnitude and it should not influence the likelhood
    # at all.
    if (age1 is None) or (age2 is None):
        return 0
    mean, std = get_age_mean_std_for_rel_tuple(rel_tuple=rel_tuple, age1=age1, age2=age2)
    age_diff = age1 - age2
    return scipy.stats.norm.logpdf(age_diff, loc=mean, scale=std)


class PwLogLike:
    """
        Class for getting pairwise log likelihoods
        on the fly.

        Usage:
            pwll = PwLogLike(
                bio_info=bio_info,
                phased_ibd_seg_list=phased_ibd_seg_list,
            )
            pwll.get_pw_ll(node1, node2, rel_tuple)
    """

    def __init__(
        self,
        bio_info : list[dict[str, int]],
        unphased_ibd_seg_list : list[
            tuple[
                int,  # id1
                int,  # id2
                str,  # chrom
                int,  # start
                int,  # end
                bool, # is full
                float,  # seg_cm
            ]
        ],
        condition_pair_set : Optional[frozenset[int]] = None,
        r : float=R,
        p : float=P,
        c : float=C,
        q : float=Q,
        mean_bgd_num : float=MEAN_BGD_NUM,
        mean_bgd_len : float=MEAN_BGD_LEN,
        min_seg_len : float=MIN_SEG_LEN,
    ):
        """
        Initialize all the data structures we'll need
        for computing pairwise likelihoods.

        Args:
            bio_info: list
                List of the form
                    [{'genotype_id' : ID, 'age' : AGE, 'sex' : SEX, 'coverage' : COVERAGE},...]
                Note: coverage is optional. Assumes full coverage if missing.
            unphased_ibd_seg_list: list
                Optional list of the form
                    [[id1, id2, chrom, phys_start, phys_end, is_full, seg_cm]]
            condition_pair_set: set
                Sed of frozensets of pairs to condition on. If a pair is in this set, use the
                conditional likelihood. If not, use the unconditional likelihood.
            r: expected number of recombinations per genome, per meiosis
            q: coverage scaling constant controling the probability of observing a segment.
                The probability of observing a segment with true length l between a person with
                coverage cov1 and a person with coverage coverage cov2 is
                    cov2 = 1 - exp(-p * cov1 * cov2 * l)
            p: coverage scaling constant controling the length of an observed segment.
                The expected length of an observed segment between two people with coverages
                cov1 and cov2, given that the true length is l is
                    l = 1 - exp(-p * cov1 * cov2) (l - \tau) + \tau
                where \tau is the minimum segment length.
            c: number of autosomes
            mean_bgd_num : mean number of background IBD segments between two unrelated people
            mean_bgd_len : mean length of a background IBD segment between two unrelated people
            min_seg_len: minimum observable segment length. Default is MIN_SEG_LEN from .constants
        """

        # set up the set of pairs to condition on
        if condition_pair_set is None:
            self.condition_pair_set = set()
        else:
            self.condition_pair_set = condition_pair_set

        # get sex, age and coverage dicts
        self.sex_dict = {
            b['genotype_id'] : b['sex']
            for b in bio_info
        }
        self.age_dict = {
            b['genotype_id'] : b['age']
            for b in bio_info
        }
        self.cov_dict = {
            b['genotype_id'] : b['coverage']
            for b in bio_info
            if 'coverage' in b  # handle legacy bio_infos without coverages
        }

        # set up constants
        self.min_seg_len = min_seg_len
        self.mean_bgd_num = mean_bgd_num
        self.mean_bgd_len = mean_bgd_len
        self.r = r
        self.p = p
        self.c = c
        self.q = q

        # load the moments of the IBD distributions
        self.cond_dict, self.uncond_dict = load_ibd_moments(min_seg_len=min_seg_len)

        # get the unphased IBD stats (num IBD1 segs, total IBD1, num IBD2 segs, total IBD2, etc...)
        self.ibd_stat_dict = get_ibd_stats_unphased(unphased_ibd_seg_list)

        # set up the pairwise relationship and pairwise log likelihood dicts
        self.pw_rels = {}
        self.pw_log_likes = {}

    # @functools.lru_cache(maxsize=None)
    def get_pw_gen_ll(
        self,
        node1 : int,
        node2 : int,
        rel_tuple : Optional[
            tuple[int, int, int]
        ],
    ):
        """
        Get the genetic component of the pairwise log
        likelihood of relationship "rel_tuple" between
        node1 and node2.

        Args:
            node1: integer node ID for person 1
            node2: integer node ID for person 2
            rel_tuple: tuple of the form (up, down, num_ancs)
                or "None" for unrelated people.

        Returns:
            pw_gen_ll: pairwise genetic component of the
                log likelihood of the relationship.
        """

        # get the key for the pair in the stat dict
        key = frozenset({node1, node2})

        # get the coverages
        cov1 = self.cov_dict.get(node1, INF)
        cov2 = self.cov_dict.get(node2, INF)

        # for easier debugging, remove "self"
        min_seg_len=self.min_seg_len
        mean_bgd_num=self.mean_bgd_num
        mean_bgd_len=self.mean_bgd_len
        q=self.q
        p=self.p

        # get the stats for the pair
        n1 = self.ibd_stat_dict[key]['num_half']
        n2 = self.ibd_stat_dict[key]['num_full']
        L1 = self.ibd_stat_dict[key]['total_half']
        L2 = self.ibd_stat_dict[key]['total_full']

        # get the moments
        moments = self.uncond_dict

        # determine whether or not to condition on observing at least one segment
        condition = key in self.condition_pair_set

        # get the PDF
        log_pdf = get_log_seg_pdf(
            n1=n1,
            n2=n2,
            L1=L1,
            L2=L2,
            rel_tuple=rel_tuple,
            cov1=cov1,
            cov2=cov2,
            min_seg_len=min_seg_len,
            mean_num_bgd=mean_bgd_num,
            mean_len_bgd=mean_bgd_len,
            moments=moments,
            p=p,
            q=q,
            condition=condition,
        )

        return log_pdf


    # @functools.lru_cache(maxsize=None)
    def get_pw_age_ll(
        self,
        node1 : int,
        node2 : int,
        rel_tuple : Optional[
            tuple[int, int, int]
        ],
    ):
        """
        Get the age component of the pairwise log
        likelihood of relationship "rel_tuple" between
        node1 and node2.

        Args:
            node1: integer node ID for person 1
            node2: integer node ID for person 2
            rel_tuple: tuple of the form (up, down, num_ancs)
                or "None" for unrelated people.

        Returns:
            pw_age_ll: pairwise age component of the
                log likelihood of the relationship.
        """
        age1 = self.age_dict[node1]
        age2 = self.age_dict[node2]
        pw_age_ll = get_age_log_like(
            age1=age1,
            age2=age2,
            rel_tuple=rel_tuple,
        )
        return pw_age_ll


    # @functools.lru_cache(maxsize=None)
    def get_pw_ll(
        self,
        node1 : int,
        node2 : int,
        rel_tuple : Optional[
            tuple[int, int, int]
        ],
    ):
        """
        Get the pairwise log likelihood of relationship
        "rel_tuple" between node1 and node2.

        Accounts for both genetics and ages.

        Args:
            node1: integer node ID for person 1
            node2: integer node ID for person 2
            rel_tuple: tuple of the form (up, down, num_ancs)
                or "None" for unrelated people.

        Returns:
            pw_ll: pairwise log likelihood of the relationship.
        """

        if get_deg(rel_tuple) > UNREL_DEG:
            rel_tuple = None

        if (
            node1 in self.pw_log_likes and
            node2 in self.pw_log_likes[node1] and
            rel_tuple in self.pw_log_likes[node1][node2]
        ):  # if this likelihood has been computed
            pw_ll = self.pw_log_likes[node1][node2][rel_tuple]
            return pw_ll
        else:  # if the likelihood has not been computed
            pw_gen_ll = self.get_pw_gen_ll(
                node1=node1,
                node2=node2,
                rel_tuple=rel_tuple,
            )
            pw_age_ll = self.get_pw_age_ll(
                node1=node1,
                node2=node2,
                rel_tuple=rel_tuple,
            )
            pw_ll = pw_gen_ll + pw_age_ll

            if node1 not in self.pw_log_likes:
                self.pw_log_likes[node1] = {}
            if node2 not in self.pw_log_likes[node1]:
                self.pw_log_likes[node1][node2] = {}
            self.pw_log_likes[node1][node2][rel_tuple] = pw_ll

            rev_rel_tuple = reverse_rel(rel_tuple)
            if node2 not in self.pw_log_likes:
                self.pw_log_likes[node2] = {}
            if node1 not in self.pw_log_likes[node2]:
                self.pw_log_likes[node2][node1] = {}
            self.pw_log_likes[node2][node1][rev_rel_tuple] = pw_ll

            return pw_ll


def coverage_corrected_mean_num_segs(
    mean_num: float,
    min_seg_len: float,
    mean_len_bgd: float,
    mean_num_bgd: float,
    cov1: float,
    cov2: float,
    m: int,
    q: float,
):
    """
    Correct the expected number of segments for coverage.

    Args:
        mean_num: expected number of segments
        min_seg_len: minimum segment length
        cov1: coverage of person 1
        cov2: coverage of person 2
        m: number of meioses separating the two people
        q: false negative parameter for low coverage genomes
           See: https://www.science.org/doi/10.1126/
                Equation S6.3

           This formula says that the probability that a segment
           is observed given that one person has coverage c is
              1 - exp(-q * c * l)
           where l is the true length of the segment. So if we
           have two people with coverages c1 and c2, then the
           probability that the segment is observed is approximately
           the product (1 - exp(-q*c1*l)) * (1 - exp(-q*c2*l)).

    Returns:
        mean_num: corrected expected number of segments
    """
    # no correction for unrelated.
    if m == INF:
        return (
            mean_num_bgd *
            (1 - np.exp(-q * cov1 * mean_len_bgd)) *
            (1 - np.exp(-q * cov2 * mean_len_bgd))
        )
    else:
        return mean_num * (1 - np.exp(-q * cov1 * (100/m))) * (1 - np.exp(-q * cov2 * (100/m)))


def get_log_seg_pdf(
    n1: int,
    n2: int,
    L1: float,
    L2: float,
    rel_tuple: tuple[int, int, int],
    cov1: float,
    cov2: float,
    min_seg_len: float,
    mean_num_bgd: float,
    mean_len_bgd: float,
    moments : dict[str, Any],
    p : Optional[float]=P,
    q : Optional[float]=Q,
    condition: bool=True,
):
    """
    Get the log PDF of the total lengths of IBD1 and IBD2
    for a particular relationship tuple rel_tuple.
    """

    # get the moments
    if rel_tuple is None:
        mean_n1 = mean_num_bgd
        mean_l1 = mean_len_bgd
        mean_L1 = mean_n1 * mean_l1
        # mean_n2 is the prob that a bgd seg overlaps an IBD1 seg (mean_L1/GENOME_LENGTH)
        # and is on other side (1/2), times num bgd segs
        mean_n2 = mean_num_bgd * (mean_L1 / GENOME_LENGTH) / 2
        # given that two segments overlap, their overlap is uniformly distributed between
        # zero and mean_len_bgd
        mean_l2 = mean_len_bgd / 2
    else:
        if (rel_tuple not in moments):
            rel_tuple = (rel_tuple[0] + rel_tuple[1], rel_tuple[2])
        mean_n1 = moments[rel_tuple]['mean_n1']
        mean_l1 = moments[rel_tuple]['mean_l1']
        mean_L1 = moments[rel_tuple]['mean_L1']
        if rel_tuple == (1,1,2):
            mean_n2 = moments[rel_tuple]['mean_n2']
            mean_l2 = moments[rel_tuple]['mean_l2']
        else:
            # set the IBD2 values for non-full-sibs to some small value
            # just in case people carry some IBD2 by chance
            mean_n2 = mean_num_bgd * (mean_L1 / GENOME_LENGTH) / 2
            mean_l2 = mean_len_bgd / 2

    # correct the mean segment lengths for coverage
    # adjust for machine precision by using 1-exp(-x) \approx x for small x
    cov_factor1 = 1 - np.exp(-p*cov1) if p*cov1 > 1e-5 else p*cov1
    cov_factor2 = 1 - np.exp(-p*cov2) if p*cov2 > 1e-5 else p*cov2
    cov_factor = cov_factor1 * cov_factor2
    mean_l1 = cov_factor * (mean_l1 - min_seg_len) + min_seg_len
    mean_l2 = cov_factor * (mean_l2 - min_seg_len) + min_seg_len

    # TODO: add the false positive segment rate due to coverage
    # to the background IBD rate.

    # correct the expected numbers of segments for coverage
    if rel_tuple is None:
        m = INF
    else:
        m = rel_tuple[0] + rel_tuple[1] if len(rel_tuple) == 3 else rel_tuple[0]
    mean_n1 = coverage_corrected_mean_num_segs(
        mean_num=mean_n1,
        min_seg_len=min_seg_len,
        mean_len_bgd=mean_len_bgd,
        mean_num_bgd=mean_num_bgd,
        cov1=cov1,
        cov2=cov2,
        m=m,
        q=q,
    )
    mean_n2 = coverage_corrected_mean_num_segs(
        mean_num=mean_n2,
        min_seg_len=min_seg_len,
        mean_len_bgd=mean_len_bgd,
        mean_num_bgd=MIN_MEAN_BGD_NUM,  # assumee few background IBD2 segs
        cov1=cov1,
        cov2=cov2,
        m=m,
        q=q,
    )

    # get the PDF for IBD1 segments
    log_pdf1 = get_n_L_log_pdf_approx(
        n=n1,
        L=L1 - n1*min_seg_len,
        mean_num=mean_n1,
        mean_len=mean_l1,
        mean_num_bgd=mean_num_bgd,
        mean_len_bgd=mean_len_bgd,
        condition=condition,
    )

    # get the PDF for IBD2 segments
    log_pdf2 = get_n_L_log_pdf_approx(
        n=n2,
        L=L2 - n2*min_seg_len,
        mean_num=mean_n2,
        mean_len=mean_l2,
        mean_num_bgd=MIN_MEAN_BGD_NUM,  # assume few background IBD2 segs
        mean_len_bgd=MIN_MEAN_BGD_LEN,  # assume few background IBD2 segs
    )

    # get the log pdf
    log_pdf = log_pdf1 + log_pdf2

    return log_pdf


def get_log_total_ibd_pdf(
    a: int,
    m: int,
    L: float,
    r: float=R,
    c: int=C,
    mean_num_bgd: float=MEAN_BGD_NUM,
    mean_len_bgd: float=MEAN_BGD_LEN,
    min_seg_len: float=MIN_SEG_LEN,
    condition: bool=False,
):
    """
    Get the log PDF of the total IBD length between two people
    separated by "m" meioses and sharing "a" common ancestors.

    Incorporate background IBD.

    Use the gamma approximtion for the sum of independent
    gamma random variables.

    Args:
        a: number of common ancestors
        m: number of meioses separating the two people. Can be a scalar,
           a list, or a 1-D numpy array.
        L: total IBD length
        r: expected number of recombination events per genome per meiosis
        c: number of autosomes
        mean_num_bgd: mean number of background segments
        mean_len_bgd: mean length of background segments
        min_seg_len: minimum observable segment length
        condition: bool. Condition on observing at least one segment.

    Returns:
        log_pdf: log pdf of the total IBD length evaluated
                 at each value of m.
    """

    # make m an array
    if isinstance(m, int):
        m = np.array([m])
    elif isinstance(m, list):
        m = np.array(m)
    elif isinstance(m, set):
        m = np.array(m)
    elif (
        (not np.isscalar(m))
        and len(m) > 0
        and isinstance(m[0], int)
    ):
        m = np.array(m)
    elif isinstance(m, np.ndarray):
        pass
    else:
        raise BonsaiError("m must be an integer, list, set, or numpy array.")

    # get the expected number of segments for each
    # value of m
    mean_num = (2.0**(1-m)) * a * (r*m+c)

    # get the expected foreground segment length
    # for each value of m
    mean_len = 1/get_lam_a_m(m_lst=m)[0]

    # handle L = 0
    if condition and L == 0:
        return -INF*np.ones(m.shape)
    elif L == 0:
        mu = mean_num + mean_num_bgd
        log_pdf = 1 - np.exp(-mu)  # probability of no observed segments
        log_pdf[mu < 1e-3] = mu[mu < 1e-3]  # approximate for small mu
        return log_pdf

    # get the list of log PDF values
    log_pdf = get_L_log_pdf_approx(
        L=L,
        mean_num=mean_num,
        mean_len=mean_len,
        mean_num_bgd=mean_num_bgd,
        mean_len_bgd=mean_len_bgd,
        min_seg_len=min_seg_len,
        condition=condition,
    )

    return log_pdf


def get_total_ibd_deg_lbd_pt_ubd(
    a: int,
    L: float,
    m: list=PW_TOTAL_IBD_M_LIST,
    alpha: float=DEG_CI_ALPHA,
    r: float=R,
    c: int=C,
    mean_num_bgd: float=MEAN_BGD_NUM,
    mean_len_bgd: float=MEAN_BGD_LEN,
    min_seg_len: float=MIN_SEG_LEN,
    condition: bool=False,
):
    """
    Infer the point estimate for the most likely number of
    meioses (m) as well as the lower and upper bounds of
    the alpha-level confidence interval on "m".

    Args:
        a: number of common ancestors
        m: number of meioses separating the two people. Can be a scalar,
           a list, set, or a 1-D numpy array.
        L: total IBD length
        alpha: confidence level (default is 0.05)
        r: expected number of recombination events per genome per meiosis
        c: number of autosomes
        mean_num_bgd: mean number of background segments
        mean_len_bgd: mean length of background segments
        min_seg_len: minimum observable segment length
        condition: bool. Condition on observing at least one segment.

    Returns:
        mlm: point estimate for the most likely number of meioses
        lbd: lower bound of the alpha-level confidence interval
        ubd: upper bound of the alpha-level confidence interval
    """

    # get the log PDF of the total IBD length L evaluated at each value of m
    log_pdf = get_log_total_ibd_pdf(
        a=a,
        m=m,
        L=L,
        r=r,
        c=c,
        mean_num_bgd=mean_num_bgd,
        mean_len_bgd=mean_len_bgd,
        min_seg_len=min_seg_len,
        condition=condition,
    )

    # normalize the log_pdf
    log_pdf = log_pdf - logsumexp(log_pdf)

    # find the most likely value of m and also the HPD interval
    # i.e., the set of values with the higheset probability mass
    # that contains 1-alpha of the total probability mass
    idx_list = np.argsort(log_pdf)[::-1]

    # get the index of the maximum value
    max_idx = idx_list[0]

    # get the indexes in the 1-alpha HPD set.
    log_cum_sum = -INF
    idx = 0
    while True:
        log_cum_sum = logsumexp(log_pdf[idx_list[:idx+1]])
        if log_cum_sum > np.log(1-alpha):
            break
        if idx == len(log_pdf)-1:
            break
        idx += 1

    # get the lower bound index
    lbd_idx = min(idx_list[:idx+1])

    # get the upper bound index
    ubd_idx = max(idx_list[:idx+1])

    # get the lower bound m, upper bound m, and most likely m
    lbd = m[lbd_idx]
    ubd = m[ubd_idx]
    mlm = m[max_idx]

    return lbd, mlm, ubd


def get_ped_like(
    ped: dict[int, dict[int, int]],
    pw_ll_cls: Any,
):
    """
    Get the composite likelihood of a pedigree
    as the sum of pairwise log likelihoods
    between each pair of genotyped IDs.

    Args:
        ped: up node dict of the form
            {node: {parent1 : deg1, parent2 : deg2}, ..}
            Can have zero, one, or two parents per node
        pw_ll_cls: instance of class with methods for computing
                   pairwise log likelihoods.

    Returns:
        log_like: float log composite likelihood of ped
    """
    gt_id_set = get_gt_id_set(ped)

    log_like = 0
    for i1,i2 in combinations(gt_id_set, r=2):
        rel_tuple = get_simple_rel_tuple(ped, i1, i2)

        ll = pw_ll_cls.get_pw_ll(node1=i1, node2=i2, rel_tuple=rel_tuple)

        log_like += ll

    return log_like

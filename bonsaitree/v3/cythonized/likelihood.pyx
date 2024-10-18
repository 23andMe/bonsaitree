# Cython-optimized functions for the segment count and total length distribution

from scipy.special import gammaln, logsumexp

import numpy as np

from itertools import combinations_with_replacement

from ..constants import R, C, MIN_SEG_LEN

INF = float('inf')


def poisson_log_pmf(k, mu):
    """
    Compute the Poisson probability mass function at k with mean mu

    Args:
        k: integer at which to evaluate the PMF
        mu: mean of the Poisson distribution

    Returns:
        Poisson PMF at k with mean mu
    """
    #return (mu**k) * exp(-mu) / gamma(k+1)
    log_pmf = k * np.log(mu) - mu - gammaln(k + 1)
    if np.isscalar(log_pmf) and mu == 0 and k == 0:
        log_pmf = 0
    elif not np.isscalar(log_pmf):
        log_pmf[(k == 0) & (mu == 0)] = 0  # handle 0 expected, 0 obtained
    return log_pmf


def gamma_log_pdf(x, a, b):
    """
    Compute the gamma probability density function at x with shape a and scale b

    Args:
        x: value at which to evaluate the PDF
        a: shape parameter of the gamma distribution
        b: scale parameter of the gamma distribution

    Returns:
        gamma PDF at x with shape a and scale b
    """
    # return (1 / (gamma(a) * b**a)) * (x**(a-1)) * exp(-x/b)
    return -gammaln(a) - a * np.log(b) + (a - 1) * np.log(x) - x / b


def get_approx_gamma_log_pdf(x, a1, a2, b1, b2):
    """
    Approximate the sum of two gamma-distributed random
    variables with parameters a1, b1 and a2, b2 by matching
    the moments of the approximating gamma distribution to
    those of the sum.

    Args:
        x: value at which to evaluate the PDF
        a1: shape parameter of the first gamma distribution
        a2: shape parameter of the second gamma distribution
        b1: scale parameter of the first gamma distribution
        b2: scale parameter of the second gamma distribution

    Returns:
        pdf: value of the PDF at x
    """
    if b2 == INF and a2 == 0:
        a = a1
        b = b1
    else:
        a = (a1*b1 + a2*b2)**2 / (a1*b1**2 + a2*b2**2)
        b = (a1*b1**2 + a2*b2**2) / (a1*b1 + a2*b2)
    return gamma_log_pdf(x=x, a=a, b=b)


def get_approx_gamma_log_pdf_shape_scale(a_arr, b_arr):
    """
    Get the shape and scale parameters of the gamma distribution
    that approximates the sum of k independent gamma-distributed random
    variables with parameters a1,...,ak and b1,...,bk. The approximating
    gamma distribution is obtained by matching the moments of the
    approximating gamma distribution to those of the sum.

    Args:
        x: value at which to evaluate the PDF
        a_arr: numpy array of shape parameters of the gamma distributions
        b_arr: numpy array of scale parameters of the first gamma distributions

    Returns:
        pdf: value of the PDF at x
    """
    a = sum(a_arr*b_arr)**2 / sum(a_arr*b_arr**2)
    b = sum(a_arr*b_arr**2) / sum(a_arr*b_arr)
    return a, b


def get_approx_gamma_log_pdf_arr(x, a_arr, b_arr):
    """
    Approximate the sum of k independent gamma-distributed random
    variables with parameters a1,...,ak and b1,...,bk by matching
    the moments of the approximating gamma distribution to
    those of the sum.

    Args:
        x: value at which to evaluate the PDF
        a_arr: numpy array of shape parameters of the gamma distributions
        b_arr: numpy array of scale parameters of the first gamma distributions

    Returns:
        pdf: value of the PDF at x
    """
    a, b = get_approx_gamma_log_pdf_shape_scale(a_arr, b_arr)
    return gamma_log_pdf(x, a, b)


def get_etas(d_arr, r=R, c=C, t=MIN_SEG_LEN):
    """
    Compute the expected number of segments from each of
    k = len(d_arr) relationships with degrees of separation
    d1,...,dk.

    Args:
        d_arr: degrees of separation d1,...,dk
        r: the expected number of recombination events
           per meiosis per genome.
        c: the total number of chromosomes
        t: the minimum observable segment length in cM

    Returns:
        etas: expected number of segments from each ancestor
    """
    return (2.0**(1-d_arr)) * (d_arr*r + c) * np.exp(-d_arr*t/100)


def get_log_prob_L_given_n_arr_d_arr(L, n_arr, d_arr, t=MIN_SEG_LEN):
    """
    Compute the log probability of observing a total length
    of L cM, given that n1,...,nk segments are observed among
    a pair of individuals related through k different lineages
    with degrees of separation d1,...,dk.

    Args:
        L: observed total length in cM
        n_arr: numpy array of observed numbers of segments
        d_arr: numpy array of degrees of separation
        t: the minimum observable segment length in cM

    Returns:
        log_pdf: log probability of observing L cM
    """
    # get the expected length of a segment from each connecting lineage
    l_arr = 100 / d_arr

    # get the total number of segments
    n = sum(n_arr)

    # adjust L to account for the fact that each segment is longer than t
    L_adj = L - n*t

    return get_approx_gamma_log_pdf_arr(x=L_adj, a_arr=n_arr, b_arr=l_arr)


def get_log_prob_num_obs(n_arr, d_arr, r=R, c=C, t=MIN_SEG_LEN):
    """
    Compute the log probability of observing n1,...,nk
    segments from k different ancestors who separate
    the observed modern individuals by degrees
    d1,...,dk.

    Args:
        n_arr: numpy array of observed numbers of segments
        d_arr: numpy array of degrees of separation
        r: the expected number of recombination events
           per meiosis per genome.
        c: the total number of chromosomes
        t: the minimum observable segment length in cM

    Returns:
        log_pdf: log probability of observing n1,...,nk segments
    """

    # get the expected number of segments from each ancestor
    e_arr = get_etas(d_arr, r, c, t)

    # compute the log probability of observing n1,...,nk segments
    return sum(
        n_arr*np.log(e_arr)
        -e_arr
        -gammaln(n_arr + 1)
    )


def get_log_prob_obs_seg(d_arr, r=R, c=C, t=MIN_SEG_LEN):
    """
    Compute the log probability of observing at least
    one segment, given that two people are connected
    by k=len(d_arr) different relationships of degrees
    (d1,...,dk).

    Args:
        d_arr: numpy array of degrees of separation
        r: the expected number of recombination events
           per meiosis per genome.
        c: the total number of chromosomes
        t: the minimum observable segment length in cM

    Returns:
        log_pdf: log probability of observing at least one segment
    """

    # get the expected number of segments from each ancestor
    e_arr = get_etas(d_arr, r, c, t)

    prob_arr = 1 - np.exp(-e_arr)
    prob_arr[e_arr < 1e-4] = e_arr[e_arr < 1e-4]  # avoid underflow

    return np.sum(np.log(prob_arr))


def get_all_vectors(k,n, min_val=0):
    """
    Get all vectors of length k that sum to n.

    Optionallly allow the minimum value of any element
    to be min_val.

    Args:
        k: int, length of vectors
        n: int, sum of vectors
        min_val: int, minimum value of each element in the vector

    Returns:
        vectors: list of all vectors of length k that sum to n
    """

    if k == 1:
        if n >= min_val:
            yield [n]
    elif k > 1:
        for i in range(min_val, n+1):
            for v in get_all_vectors(k-1, n-i, min_val):
                yield [i] + v


def get_n_L_log_pdf_approx_given_d_arr(
    n,
    L,
    d_arr,
    r=R,
    c=C,
    t=MIN_SEG_LEN,
):
    """
    Compute the probability of observing n segments
    with a total length of L cM, given that the two
    individuals are connected by k=len(d_arr) different
    relationships of degrees (d1,...,dk).

    Args:
        n: observed number of segments
        L: observed total length
        d_arr: numpy array of degrees of separation
        r: the expected number of recombination events
           per meiosis per genome.
        c: the total number of chromosomes
        t: the minimum observable segment length in cM

    Returns:
        log_pdf: log probability of observing n segments
                 with total length L
    """

    # sum over all possible vectors n_arr of observed numbers of segments
    # from each of the k relationships to get the unconditional PDF of L
    uncond_log_pdf = -INF
    k = len(d_arr)  # get the number of relationships k
    for n_arr in get_all_vectors(k, n, min_val=1):
        n_arr = np.array(n_arr)
        uncond_log_pdf = np.logaddexp(
            uncond_log_pdf,
            get_log_prob_L_given_n_arr_d_arr(L, n_arr, d_arr, t)
            + get_log_prob_num_obs(n_arr, d_arr, r, c, t)
        )

    # convert to a conditional PDF by subtracting the log probability
    # of observing at least one segment
    cond_log_pdf = uncond_log_pdf - get_log_prob_obs_seg(d_arr, r, c, t)

    return cond_log_pdf


def get_deg_to_num_ancs(d_arr):
    """
    Get a dict mapping each degree of separation in d_arr
    to the number of elements of d_arr that have that degree.

    Since we're cythonizing, we'll write our own version
    of this function instead of using a built-in one.

    Args:
        d_arr: numpy array of degrees of separation

    Returns:
        deg_to_num_ancs: dict mapping each degree of separation to the
                  number of elements of d_arr that have that degree
    """
    deg_to_num_ancs = {}
    for d in d_arr:
        if d in deg_to_num_ancs:
            deg_to_num_ancs[d] += 1
        else:
            deg_to_num_ancs[d] = 1
    return deg_to_num_ancs


def get_d_arr_log_like_list_for_n_L(
    n,
    L,
    max_d,
    r=R,
    c=C,
    t=MIN_SEG_LEN,
    exp_num_ancs_per_gen=None,
    max_num_ancs=None,
):
    """
    Compute the log likelihood of each possible number of
    connecting lineages k from 1 to n and the degree of each
    connecting lineage d from 1 to max_d.

    Return a list of tuples, where each tuple contains the
    vector d_arr = (d1,...,dk) and the log likelihood of
    observing n segments with total length L, given d_arr.

    Args:
        n: observed number of segments
        L: observed total length of IBD in cM
        max_d: maximum degree of separation on each
            lineage connecting the two individuals
        r: the expected number of recombination events
              per meiosis per genome.
        c: the total number of chromosomes
        t: the minimum observable segment length in cM
        exp_num_ancs_per_gen: list of length at least ceil(max_d/2) containing the
            expected number of IBD-transmitting
            ancestors in each generation. If None, do not weight
            by the probability of observing each number of ancestors
            in each generation. For simplicity, assume that the two
            putative relatives live in the same generation.
        max_num_ancs: maximum number of distinc ancestors to consider.
            If None, consider all possible numbers of ancestors from
            1 to n (number of segments).

    Retruns:
        log_like_list: list of tuples [(d_arr, log_like)], where each
            tuple contains the vector d_arr = (d1,...,dk) and the log
            likelihood of observing n segments with total length L,
            given d_arr.
    """
    # get the maximum number of ancestors to consider
    max_num_ancs = n if max_num_ancs is None else max_num_ancs

    log_like_list = []
    for k in range(1, max_num_ancs+1):
        for d_list in combinations_with_replacement(range(1, max_d), r=k):
            d_arr = np.array(d_list)
            log_like = get_n_L_log_pdf_approx_given_d_arr(n, L, d_arr, r=r, c=c, t=t)

            # weight by the probability of observing each number of ancestors
            log_wt = 0
            if exp_num_ancs_per_gen is not None:
                d_to_ct = get_deg_to_num_ancs(d_arr)
                g_to_ct = {np.ceil(d/2) : c for d,c in d_to_ct.items()}
                for idx, exp_ct in enumerate(exp_num_ancs_per_gen):
                    g = idx + 1
                    ct = g_to_ct.get(g, 0)
                    log_wt +=  poisson_log_pmf(k=ct, mu=exp_ct)
            log_like += log_wt

            log_like_list.append((d_arr, log_like))

    return log_like_list


def get_n_L_log_pdf_approx(
    n,
    L,
    mean_num,
    mean_len,
    mean_num_bgd,
    mean_len_bgd,
    condition=False,
):
    """
    Use the moment-matched gamma approximation to
    get the approximate joint n,L PDF.

    Args:
        n: observed number of segments
        L: observed total length
        mean_num: mean number of segments
        mean_len: mean length of a segment
        mean_num_bgd: mean number of background segments
        mean_len_bgd: mean length of a background segment
        condition: bool. Condition on observing at least one segment.

    Returns:
        log_pdf: log probability density at (n, L)
    """

    # if no segments, the pdf is just the probability of no segments.
    if n == 0 and L == 0:
        return poisson_log_pmf(0, mean_num + mean_num_bgd)

    if mean_len_bgd == 0 or mean_num_bgd == 0:
        nf = n
        if nf == 0 or L == 0:
            return -INF
        gam_pdf = gamma_log_pdf(L, nf, mean_len)
        pois_pdf = poisson_log_pmf(nf, mean_num)
        log_pdf = gam_pdf + pois_pdf
    else:
        log_pdf_list = []
        for i in range(n):  # not n+1 because we want to see at least one segment from a foreground individual. The idea is that there are no background IBD segments. # noqa E800
            nf = n - i  # number of foreground segments
            nb = i  # number of background segments
            gam_pdf = get_approx_gamma_log_pdf(L, nf, nb, mean_len, mean_len_bgd)
            pois_pdf = poisson_log_pmf(nf, mean_num) + poisson_log_pmf(nb, mean_num_bgd)
            log_pdf_list.append(gam_pdf + pois_pdf)
        log_pdf = logsumexp(log_pdf_list, axis=0)

    # condition on observing at least one segment from a non-background individual
    if condition:
        mu = mean_num  # + mean_num_bgd  # commented out + mean_num_bgd because we want to see the segment from a foreground individual # noqa E800
        cond_factor = 1 - np.exp(-mu)
        if np.isscalar(cond_factor) and mu < 1e-4:
            cond_factor = mu
        elif not np.isscalar(cond_factor):
            cond_factor[mu < 1e-4] = mu[mu < 1e-4]  # avoid underflow
        log_pdf -= np.log(cond_factor)

    return log_pdf


def get_L_log_pdf_approx(
    L,
    mean_num,
    mean_len,
    mean_num_bgd,
    mean_len_bgd,
    min_seg_len,
    max_n=100,
    condition=False,
):
    """
    Sum over all possible numbers of segments to get the
    marginal PDf of the total length.

    Use the moment-matched gamma approximation to
    get the approximate joint n,L PDF.

    Args:
        L: observed total length
        mean_num: mean number of segments
        mean_len: mean length of a segment
        mean_num_bgd: mean number of background segments
        mean_len_bgd: mean length of a background segment
        min_seg_len: minimum segment length
        max_n: maximum number of segments to sum over
        condition: bool. Condition on observing at least one segment.

    Returns:
        log_pdf: log probability density at L
    """

    log_pdf = -INF
    for n in range(1,max_n):
        n_L_log_pdf = get_n_L_log_pdf_approx(
            n=n,
            L=L-n*min_seg_len,
            mean_num=mean_num,
            mean_len=mean_len,
            mean_num_bgd=mean_num_bgd,
            mean_len_bgd=mean_len_bgd,
            condition=condition,
        )
        log_pdf = np.logaddexp(log_pdf, n_L_log_pdf)

    return log_pdf

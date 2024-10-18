from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln, logsumexp

from .constants import (
    GENOME_LENGTH,
    INF,
    KING_BD_EPS,
    MAX_KING_DEG,
    MIN_SEG_LEN,
    C,
    R,
)
from .ibd import get_total_ibd_between_id_sets
from .imports import load_total_ibd_bounds
from .pedigrees import (
    get_leaf_set_for_down_dct,
    get_subdict,
    re_root_up_node_dict,
    reverse_node_dict,
    trim_to_proximal,
)


def get_total_ibd_bds_conditional():
    bd_df = load_total_ibd_bounds()
    bds = bd_df['Conditional'].to_numpy()
    bds /= GENOME_LENGTH
    bds = [INF] + [*bds]
    bds[1] = 0.95  # adjust parent/child bound up
    return np.array(bds)


def get_total_ibd_bds_unconditional():
    bd_df = load_total_ibd_bounds()
    bds = bd_df['Unconditional'].to_numpy()
    bds /= GENOME_LENGTH
    bds = [INF] + [*bds]
    bds[1] = 0.95  # adjust parent/child bound up
    return np.array(bds)


def get_king_bds(
    max_deg : int=MAX_KING_DEG,
    king_bd_eps: float=KING_BD_EPS,
):
    """
    Get boundaries of fraction of genome shared
    for different relative degrees. For all degrees
    from 0 to max_deg, compute the total fraction
    of the genome that is shared up to degrees of d+1/2.

    Args:
        max_deg: maximum degree
        king_bd_eps:
            if people are "self", then they will share the whole genome identically
            the king bounds set people to "self" if they share more than ~70% of the
            genome. we should set this bound almost to 1. king_bd_eps defines "almost."

    Returns:
        bds: bds[d] is the total fraction of the genome
             shared between two people separated by degree
             d + 1/2
    """
    deg_arr = np.arange(0, max_deg)
    expon = deg_arr + 1/2
    bds = 2**(-expon)
    bds = 2*bds

    # Adjust the thresholds for calling someone the same person
    bds[0] = 2
    bds[1] = 1 - king_bd_eps

    return bds


def get_log_mu_amt(
    m_arr : npt.ArrayLike,
    a : int=1,
    r : float=R,
    c : int=C,
    min_seg_len : int=MIN_SEG_LEN,
):
    """
    Get the log value of the mean number of segments shared between
    two people separated by m meioses and a common ancestors.

    Args:
        m_arr: numpy array of numbers of meioses
        a: number of common ancestors (1 or 2)
        r: expected number of recombinations per genome per meiosis
        c: number of autosomes
        min_seg_len: minimum observable segment length

    Returns:
        log_mu_amt: log of the expected number of shared segments
                    at each value of m_arr
    """
    log_mu_amt = (1-m_arr)*np.log(2) + np.log(a) +  np.log(r*m_arr+c) - m_arr*min_seg_len/100
    log_mu_amt[m_arr == 0] = np.log(2*c)
    log_mu_amt[m_arr == 1] = np.log(c)
    return log_mu_amt


def get_mu_amt(
    m_arr : npt.ArrayLike,
    a : int=1,
    r : float=R,
    c : int=C,
    min_seg_len : int=MIN_SEG_LEN,
):
    """
    Get the mean number of segments shared between
    two people separated by m meioses and a common ancestors.

    Args:
        m_arr: numpy array of numbers of meioses
        a: number of common ancestors (1 or 2)
        r: expected number of recombinations per genome per meiosis
        c: number of autosomes
        min_seg_len: minimum observable segment length

    Returns:
        mu_amt: expected number of shared segments
                at each value of m_arr
    """
    mu_amt = (2**(1-m_arr)) * a * (r*m_arr+c) * np.exp(-m_arr*min_seg_len/100)
    mu_amt[m_arr == 0] = 2*c
    mu_amt[m_arr == 1] = c
    return mu_amt


def get_lam_am(
    m_arr : npt.ArrayLike,
    a : int=1,
    r : float=R,
    c : int=C,
    genome_len: float=GENOME_LENGTH,
):
    """
    Get inverse expected length of a shared segment between
    two people separated by m meioses and a common ancestors.

    Args:
        m_arr: numpy array of numbers of meioses
        a: number of common ancestors (1 or 2)
        r: expected number of recombinations per genome per meiosis
        c: number of autosomes

    NOTE: there is no min_seg_len because the exponential distribution
          is memoryless. #TODO: make sure you believe this.

    Returns:
        lam_amt: inverse expecte length of a segment.
    """
    lam_amt = m_arr/100
    lam_amt[m_arr <= 1] = c/genome_len
    return lam_amt


def get_conditional_king_bds(
    max_deg : int=MAX_KING_DEG,
    r : float=R,
    c : int=C,
    min_seg_len : int=MIN_SEG_LEN,
    king_bd_eps: float=KING_BD_EPS,
):
    """
    Get boundaries of fraction of genome shared
    for different relative degrees, conditonal on observing
    any IBD at all. For all degrees from 0 to max_deg,
    compute the total fraction of the genome that is shared
    up to degrees of d+1/2.

    This is given by the formula:

        total_shared / 2*GENOME_LENGTH

    where

        total_shared = (min_seg_len + 1/lam_amt) * mu_amt / (1 - np.exp(-mu_amt))

    Here, lam_amt is the inverse expected length of a shared segment. The
    total shared is the expected number of shared segments times the expected
    length of a shared segment. Because every segment has minimum length min_seg_len
    and because the exponential distribution is memoryless, we add 1/lam_amt to the
    minimum segment length to get the expected length of a shared segment. We then
    multiply by the expected number of shared segments and divide by the probability
    that the number of shared segments is greater than zero.

    # NOTE: This basically gives the same bounds as the root-finding approach.

    Args:
        max_deg: maximum degree
        r: the expected number of recombinations per genome per meiosis
        c: number of autosomes
        min_seg_len: minimum observable segment length in cM
        king_bd_eps: float
            if people are "self", then they will share the whole genome identically
            the king bounds set people to "self" if they share more than ~70% of the
            genome. We should set this bound almost to 1. king_bd_eps defines "almost."

    Returns:
        bds: bds[d] is the total fraction of the genome
             shared between two people separated by degree
             d + 1/2
    """
    a = 1
    m_arr = np.arange(0, max_deg) + 1/2

    # get the inverse expected segment length and the expected number of segments
    lam_am = get_lam_am(m_arr, a=a, r=r, c=c)
    log_mu_amt = get_log_mu_amt(m_arr, a=a, r=r, c=c, min_seg_len=min_seg_len)

    # compute the log of the total shared, which is the expected number of segments
    # times the expected length of a segment divided by the probability that the number
    # of segments is greater than zero. I.e.,
    #   bds = (min_seg_len + 1/lam_amt) * mu_amt / (1 - np.exp(-mu_amt)) / (2*GENOME_LENGTH)
    log_term = np.log(1 - np.exp(-np.exp(log_mu_amt)))
    log_term[log_term == -INF] = log_mu_amt[log_term == -INF]  # handle underflow in 1 - e^{-z}
    log_bds = log_mu_amt + np.log(min_seg_len + 1/lam_am) - log_term - np.log(2*GENOME_LENGTH)
    bds = np.exp(log_bds)

    # Adjust the thresholds for calling someone the same person
    bds[0] = 2
    bds[1] = 1 - king_bd_eps

    return bds


def get_conditional_druid_log_like(
    L : float,
    m : int,
    a : int,
    r : float=R,
    c : int=C,
    min_seg_len : int=MIN_SEG_LEN,
):
    """
    Get the approximate log likelihood of a particular
    DRUID connection by treating the total IBD shared
    as a Gamma random variable with k given by the expected
    number of shared segments between the ancestors and
    beta given by the inverse expected length of a shared
    segment between the ancestors.

    Args:
        L : total inferred IBD shared between two people
        m : number of meioses
        a : number of shared ancestors
        r: the expected number of recombinations per genome per meiosis
        c: number of autosomes
        min_seg_len: minimum observable segment length in cM.

    Returns:
        log_pdf: This is an approximate likelihood obtained
                 by assuming that the number of segments we observe
                 is the expected number for the relationship (m,a)
                 and that the segment length is similarly the
                 expected length. If this is the case, then the
                 likelihood of the observed segment length is

                    Gamma(E[n], E[beta])(L) * Poisson(lam_am)(E[n]) * 1/(1 - exp(-mu_am))

                This comes from the fact that the true PDF of the
                total IBD length is a sum of gammas:

                    f_L(l|O;a,m)
                        = \\sum_{n=1}^\\infty f_L(l|n,O;a,m) Pr(n|O;a,m)
                        = \\sum_{n=1}^\\infty \frac{\beta^n}{\\Gamma(n)} l^{n-1} e^{-\beta l} Pr(n|O;a,m)
                        = \\sum_{n=1}^\\infty \frac{\beta^n}{\\Gamma(n)} l^{n-1} e^{-\beta l} \frac{\\mu_{a,m} e-{\\mu_{a,m}}}{\\Gamma(n+1)} \frac{1}{1-e^{-\\mu_{a,m}}}

                And we make the approximation that all terms are small except the
                one corresponding to the expected number and length of segments
                for the relationship (m,a)
    """  # noqa: E501
    m_arr = np.array([m]).astype(float)
    lam_am = get_lam_am(m_arr, a=a, r=r, c=c)
    mu_amt = get_mu_amt(m_arr, a=a, r=r, c=c, min_seg_len=min_seg_len)

    # get the gamma part of the approximate likelihood
    gamma_log_pdf = mu_amt*np.log(lam_am) + (mu_amt-1)*np.log(L) - lam_am*L - gammaln(mu_amt)

    # get the Poisson component
    poisson_log_pmf = (
        mu_amt*np.log(mu_amt) -
        mu_amt -
        gammaln(mu_amt + 1) -
        np.log(1 - np.exp(-mu_amt))
    )

    return gamma_log_pdf + poisson_log_pmf


def get_ibd_pattern_log_prob(
    node: int,
    node_dict: dict[int, dict[int, int]],
    ibd_presence_absence_dict: dict[int, bool],
) -> tuple[float, float]:
    """
    Compute a tuple, (n0,n1), where n0 is the probability of observing
    the haplotype presence-absence pattern in the leaves descended from
    'node' if the haplotype was not transmitted as far as node and n1 is
    the probability of observing the haplotype presence-absence pattern
    in the leaves descended from node if the haplotype was transmitted
    as far as node. See Presentations >> RelationshipInference_7_12_2019
    for a derivation.

    Args:
        node_dict : dict
            A dict of the form
                { node : {desc1 : deg1, desc2 : deg2, ....} }
            node_dict skips nodes that are ancestral to only one person.
        ibd_presence_absence: dict
            Dictionary of the form
                {leaf_i : o_i, ...},
            where o_i == True if the haplotype is observed in leaf_i and
            o_i == False, otherwise. Typically the keys of ibd_presence_absence_dict
            are leaves, but they can be any independent set of nodes, which, if the
            tree were truncated at them would comprise the full leaf set at that level
            of the tree.
        node : int
            id of the node whose descendants correspond to the keys in
            ibd_presence_absence_dict.
    """

    # If the node is a leaf, or if the node is its own parent (so it's an orphan)
    if (node not in node_dict) or ((node in node_dict[node]) and (len(node_dict[node]) == 1)):
        state = ibd_presence_absence_dict.get(node, False)
        if state:
            return (-INF, 0)
        else:
            return (0, -INF)

    desc_node_dict = node_dict[node]
    n0 = 0.0
    n1 = 0.0
    for desc_node, g in desc_node_dict.items():
        desc_log_prob = get_ibd_pattern_log_prob(desc_node, node_dict, ibd_presence_absence_dict)
        if g > 0:
            log_trans_prob = np.log(1 - 2 ** (-g))
        else:
            log_trans_prob = -INF
        n0 += desc_log_prob[0]
        n1 += logsumexp(
            [desc_log_prob[0] + log_trans_prob, desc_log_prob[1] - g * np.log(2)]
        )
    return (n0, n1)


def get_T_and_leaf_set(
    down_dct : dict[int, dict[int, int]],
    anc_id : int,
):
    """
    Find the fraction of the genome of anc_id that is
    passed to the leaf nodes in node_dict

    Args:
        down_dct: node dict of the form
                   { node : {desc1 : deg1, desc2 : deg2, ....} }
                   generally rooted at anc_id/partner_id, but not
                   necessarily if we just want the fraction passed
                   from anc_id to their descendants where anc_id
                   is part of a larger down node dict.
        anc_id: node whose descendants we want

    Returns:
        T: float. Fraction of the genome that is passed to
           the most proximal genotyped descendants.
        leaf_set: the set of leaves at which the transmission
                  pattern was computed.
    """

    trim_dct = trim_to_proximal(
        down_dct=down_dct,
        root=anc_id,
    )

    leaf_set = get_leaf_set_for_down_dct(trim_dct)

    pres_abs_dict = {uid : False for uid in leaf_set}

    log_prob_tuple = get_ibd_pattern_log_prob(
        node=anc_id,
        node_dict=trim_dct,
        ibd_presence_absence_dict=pres_abs_dict,
    )

    T = 1 - np.exp(log_prob_tuple[1])

    return T, leaf_set


def get_proximal_and_shared(
    up_dct : dict[int, dict[int, int]],
    iid : int,
    dir : int,
):
    """
    Find the set (id_set) of genotyped IDs who are most
    proximal to a focal ID (iid) as well as
    the expected fraction of iid's genome that they share
    with id_set.

    Args:
        up_dct: up node dict for a pedigree of the form { node : {parent1 : deg1, parent2 : deg2} }
        iid: focal individual
        dir: direction of the connection:
                0: join down from node
                1: join up from node

    Returns:
        id_set: set of genotyped relatives that are most proximal
                to iid. For any two genotyped people (A,B) in up_dct
                who are related to iid: if A is related to iid only
                through B then return only B.
        frac: expected fraction of iid's genome that they share
              with id_set.
    """
    if iid > 0:
        frac = 1
        id_set = {iid}
        return frac, id_set

    if dir == 1:  # joining up from node. Most proximal IDs are all descendants
        down_dct = reverse_node_dict(up_dct)
        r_dct = get_subdict(
            dct=down_dct,
            node=iid,
        )
    elif (dir == 0) or (dir is None): # joining down from node. All proximal IDs are acceptable.
        # re-root up_dct to iid
        r_dct = re_root_up_node_dict(
            up_dct=up_dct,
            node=iid,
        )

    # get the most proximal relatives of iid (id_set)
    # and the fraction of iid's genome that they
    # share with id_set.
    frac, id_set = get_T_and_leaf_set(
        down_dct=r_dct,
        anc_id=iid,
    )

    return frac, id_set


# TODO: memoize this.
def infer_degree_generalized_druid(
    anc_id1 : int,
    anc_id2 : int,
    partner_id1 : int,
    partner_id2 : int,
    dir1 : int,
    dir2 : int,
    up_dct1 : dict[int, dict[int, int]],
    up_dct2 : dict[int, dict[int, int]],
    ibd_seg_list : tuple[int, int, int, int, str, float, float, float],
    condition : Optional[bool]=False,
    min_seg_len : Optional[float]=MIN_SEG_LEN,
) -> int:
    """
    Use our generalized version of the DRUID estimator to infer the
    degree separating the ancestors of node_dict1 and node_dict2
    Args:
        anc_id1: int
            ancestor in pedigree 1 through which nodes are connected to pedigree 2
        anc_id2: int
            ancestor in pedigree 2 through which nodes are connected to pedigree 1
        partner_id1: int
            partner of ancestor in pedigree 1 through which nodes are connected to pedigree 2
        partner_id2: int
            partner of ancestor in pedigree 2 through which nodes are connected to pedigree 1
        dir1: direction anc_id1 connects to anc_id2:
                0: down from anc_id1
                1: up from anc_id1
        dir2: direction anc_id2 connects to anc_id1:
                0: down from anc_id2
                1: up from anc_id2
        up_dct1: dict
            up node dict for the first pedigree : a dict of the form
                { node : {parent1 : deg1, parent2 : deg2} }
        up_dct2: dict
            up node dict for the second pedigree : a dict of the form
                { node : {parent1 : deg1, parent2 : deg2} }
        ibd_seg_list: list
            list of the form
                [[id1, id2, hap1, hap2, chromosome, start, end, seg_cm]]
        condition: bool.
            If true, condition on observing at least some IBD between the ancestors
        min_seg_len: float
            minimum observable segment length

    Returns: The degree estimate between
    """

    # get the set of genotyped IDs that are
    # most proximal to anc_id1 and the fraction
    # of anc_id1's genome they share with this set.
    frac1, id_set1 = get_proximal_and_shared(
        up_dct=up_dct1,
        iid=anc_id1,
        dir=dir1,
    )

    # get the set of genotyped IDs that are
    # most proximal to anc_id2 and the fraction
    # of anc_id2's genome they share with this set.
    frac2, id_set2 = get_proximal_and_shared(
        up_dct=up_dct2,
        iid=anc_id2,
        dir=dir2,
    )

    # if partner_id1, get the set of its most proximal genotyped
    # relatives and the fraction of its genome that it shares with them
    if partner_id1 is None:
        p_frac1 = 0
        p_id_set1 = set()
    else:
        p_frac1, p_id_set1 = get_proximal_and_shared(
            up_dct=up_dct1,
            iid=partner_id1,
            dir=dir1,
        )

    # if partner_id2, get the set of its most proximal genotyped
    # relatives and the fraction of its genome that it shares with them
    if partner_id2 is None:
        p_frac2 = 0
        p_id_set2 = set()
    else:
        p_frac2, p_id_set2 = get_proximal_and_shared(
            up_dct=up_dct2,
            iid=partner_id2,
            dir=dir2,
        )

    # merge ID sets
    id_set1 = id_set1 | p_id_set1
    id_set2 = id_set2 | p_id_set2

    # FIXFIX: [EDITED 2024-03-30] this has been fixed perhaps
    #         as much as it is possible to fix it by merging
    #         segments on the correct haplogytpe if id_set1
    #         or id_set2 has exactly one member. If there
    #         is more than one person in either of these sets
    #         then currently IBD phasing sufficiently uncertain
    #         to make it difficult to merge segments on any given
    #         haplotype with certainty. In the future, we may
    #         be able to do this with improved phasing, or by
    #         leveraging the topology as we build to phase segments.
    #
    #         This merges all segments assuming they
    #         are on the same haplotype. We actually
    #         need to account for phased IBD and count
    #         up the total IBD shared across both haplotypes.
    #         See the note on get_total_ibd_between_id_sets()
    #         However, we probably don't know what side the IBD
    #         is on until we infer the relationship (unless we
    #         pre-infer the side using some kind of triangulation
    #         or something). So perhaps we can rely on simply having
    #         wiggle room introduced by searching over the nearby
    #         degrees. NOTE: it is GENERALLY true that the phase of
    #         the segments only matters for first degree relationships
    #         because these are the ones in which segments typically
    #         occurr on more than one haplotype. However, segment phase
    #         can matter for more distant relationships as well, such as
    #         the descendants of two full sibs (the descendants can inherit
    #         IBD2 segs as in the following diagram):
    #
    #                   (-1) ----- (-2)
    #                        ___|___
    #                        |      |
    #                        1      2
    #           segs -->    | |    | |
    #                      |   |  |   |
    #                      A   B  C   D
    #
    #       The IBD2 segments can be passed down to different descendants
    #       (A, B, C, D) of 1 and 2. The segments that A,B share with C,D
    #       satisfy A IBD to C and B IBD to D so the phase does matter and
    #       we can't just squash the segment in A with the segment in B.
    L_union = get_total_ibd_between_id_sets(
        id_set1=id_set1,
        id_set2=id_set2,
        ibd_seg_list=ibd_seg_list,
    )

    # find the fraction of anc_id1's genome (or anc_id1 + partner_id1 if it has a partner)
    # that is shared with its most proximal genotyped relatives
    # The shared fraction is 1 - Pr(S^c), where S^c is the event that a given allele is not shared
    # if the two ancestors are labelled A and B, we can find the probability that an allele is not
    # shared by conditioning on whether it originated in A or B.
    #   Pr(S^c) = Pr(S^c | A) Pr(A) + Pr(S^c | B) Pr(B)
    #         = Pr(S^c | A)/2 + Pr(S^c | B)/2
    #         = [(1-Pr(S|A)) + (1-Pr(S|B))]/2
    # so
    #   Pr(S) = 1 - [(1-Pr(S|A)) + (1-Pr(S|B))]/2
    if partner_id1 is None:
        scale_factor1 = frac1
    else:
        scale_factor1 = 1 - ((1-frac1) + (1-p_frac1))/2
    if partner_id2 is None:
        scale_factor2 = frac2
    else:
        scale_factor2 = 1 - ((1-frac2) + (1-p_frac2))/2

    # estimate the total IBD shared between (anc_id1, partner_id1)
    # and (anc_id2, partner_id2). Note: anc_id1 and anc_id2 can only
    # both have partners when they are both the same people.
    L_est = L_union / (scale_factor1 * scale_factor2)

    # get the estimated fraction of the genome that (anc_id1, partner_id1)
    # shares with (anc_id2, partner_id2)
    if (partner_id1 is not None) and (partner_id2 is not None):
        genome_factor = 4
    else:
        genome_factor = 2
    F_est = L_est / (genome_factor * GENOME_LENGTH)

    if condition:
        king_bds = get_conditional_king_bds(
            max_deg=100,
            r=R,
            c=C,
            min_seg_len=min_seg_len,
        )
    else:
        king_bds = get_king_bds(max_deg=100)

    est_deg = sum(king_bds > F_est) - 1

    # if we are connecting a person with an ancestral pair
    # then the amount shared with the ancestral pair at degree d
    # is the amount shared with a single member of that pair at
    # degree d+1. So if it's an ancestral pair, we need to
    # adjust by one degree.
    if (partner_id1 is not None) or (partner_id2 is not None):
        est_deg += 1

    return max(0,est_deg), L_est

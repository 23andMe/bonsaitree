import os

INF = float('inf')

MIN_SEG_LEN = 0

# genetic map parameters
UNREL_DEG = (INF,INF,None)
AUTO_GENOME_LENGTH = 3545
FULL_GENOME_LENGTH = 3725
GENOME_LENGTH = AUTO_GENOME_LENGTH
R = 35.5  # expected number of recombination events per human genome, per meiosis  # noqa E501
C = 22  # number of human autosomes  # noqa E501

# low coverage parameters
P = 1.79  # parameter for the length scaling of low coverage segments  # noqa E501
Q = 0.179  # parameter of the false negative rate for low coverage segments  # noqa E501

# druid parameters
MAX_KING_DEG = 100
KING_BD_EPS = 5e-2  # if two genomes share a fraction 1-KING_BD_EPS IBD, then we call these people the same person  # noqa E501

# bonsai build parameters
MAX_PEDS = 3  # maximum number of pedigrees to explore at each round.  # noqa E501
MAX_START_PEDS = 3  # maximum number of pedigrees to use as starting points for the search of MAX_PEDS pedigrees.  # noqa E501
DEG_DELTA = 1  # consider all degrees at most DEG_DELTA from the DRUID point estimate  # noqa E501
MAX_CON_PTS = INF  # maximum number of connection points to consider when combining two pedigrees  # noqa E501
RESTRICT_CON_PTS = True  # restrict the set of connection points to those that are nodes in the tree connecting the set of genotyped nodes in each pedigree  # noqa E501
CONNECT_UP_ONLY = False  # only connect two sub pedigrees upward through their common ancestors  # noqa E501
TWIN_THRESHOLD = 0.95 * GENOME_LENGTH  # if two genomes share more than TWIN_THRESHOLD IBD, then we call them twins if their sexes and ages match  # noqa E501
MAX_UP_IBD=500  # when connect_up_only is specified, connect up only if the closest IDs share less than MAX_UP_IBD cM.  # noqa E501

# background IBD parameters
MEAN_BGD_NUM = 0.01  # expected number of background/false-positive segments  # noqa E501
MEAN_BGD_LEN = 5  # expected length of a background/false-positive segment if it occurs  # noqa E501
MIN_MEAN_BGD_NUM = 0.01  # minimum expected number of background/false-positive segments. Used to keep the expectation from being identically zero  # noqa E501
MIN_MEAN_BGD_LEN = 0.01  # minimum expected length of a background/false-positive segment. Used to keep the expectation from being identically zero  # noqa E501

# directory paths
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "test", "ped_fixture_data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
V3_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
AGE_MOMENT_FP = os.path.join(MODELS_DIR, 'age_diff_moments.json')
IBD_MOMENT_FP = os.path.join(MODELS_DIR, 'distn_dict.json')
IBD011_MOMENT_FP = os.path.join(MODELS_DIR, 'ibd011_moments.json')
GENETIC_MAP_FILE = os.path.join(MODELS_DIR, 'ibd64_metadata_dict.json')
LBD_PATH = os.path.join(V3_MODELS_DIR, "total_ibd_length_bounds.csv")

# relationship parameters
UNREL_UP = 250
UNREL_DEG = 2*UNREL_UP  # must be divisible by 2  # noqa E501
UNREL_TUPLE = (UNREL_UP, UNREL_UP, 1)  # rel tuple corresponding to being effectively unrelated  # noqa E501

# pairwise likelihood parameters
DEG_CI_ALPHA = 0.05  # confidence level for confidence intervals on the number of meioses separating two people with "a" common ancestors  # noqa E501
PW_TOTAL_IBD_M_LIST = [*range(1, 500)]  # list of numbers of meioses to consider when inferring the number of meioses separating two people, given pairwise total IBD  # noqa E501

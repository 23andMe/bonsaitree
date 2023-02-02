import copy
import gzip
import json
import logging
import os
from collections import defaultdict
from itertools import combinations, product

import numpy as np
import pytest
import random

from ..bonsaitree import utils
from ..bonsaitree import distributions
from ..bonsaitree import point_predictor

from ..bonsaitree.bonsai import build_pedigree
from ..bonsaitree.copytools import deepcopy
from ..bonsaitree.pedigree_object import PedigreeObject, merge_pedigrees_on_founder, extend_up, extend_down, connect_pedigrees_through_founders
from ..bonsaitree.exceptions import *
from ..bonsaitree.analytical_likelihood_tools import *
from ..bonsaitree.connect_pedigree_tools import *
from ..bonsaitree.node_dict_tools import *
from ..bonsaitree.ibd_tools import *
from ..bonsaitree.build_pedigree import *
from ..bonsaitree.distributions import AUTO_GENOME_LENGTH as GENOME_LENGTH

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

INF = float('inf')

log = logging.getLogger(__name__)



# ================== Test utils

def test_transform_segment_lists_to_ibd_summaries():
    ibd_segment_list = [
        [1, 2, "8", 80660500, 146364022, False, 71.77],
        [1, 2, "9", 2936634, 141213431, False, 157.76],
        [1, 2, "1", 2927363, 5133431, True, 6.75],
        [1, 2, "1", 18398381, 68631622, True, 61.21],
    ]

    summaries = utils.transform_segment_lists_to_ibd_summaries(ibd_segment_list)
    
    key = frozenset({1,2})
    rounded_summaries = {k : np.floor(v) for k,v in summaries[key].items()}
    assert rounded_summaries == {
        "total_half": np.floor(71.77 + 157.76),
        "total_full": np.floor(6.75 + 61.21),
        "num_half": 2,
        "max_seg_cm": np.floor(157.76),
    }


def test_transform_segment_dicts_to_ibd_summaries():
    ibd_segment_dicts = [
        {
            "genotype_id_1": 1,
            "genotype_id_2": 2,
            "chromosome": "8",
            "start": 80660500,
            "end": 146364022,
            "is_full_ibd": False,
            "seg_cm": 73.667546,
        },
        {
            "genotype_id_1": 1,
            "genotype_id_2": 2,
            "chromosome": "9",
            "start": 2936634,
            "end": 141213431,
            "is_full_ibd": False,
            "seg_cm": 151.828402,
        },
        {
            "genotype_id_1": 1,
            "genotype_id_2": 2,
            "chromosome": "1",
            "start": 2927363,
            "end": 5133431,
            "is_full_ibd": True,
            "seg_cm": 7.544491000000001,
        },
        {
            "genotype_id_1": 1,
            "genotype_id_2": 2,
            "chromosome": "1",
            "start": 18398381,
            "end": 68631622,
            "is_full_ibd": True,
            "seg_cm": 59.876436,
        },
    ]

    summaries = utils.transform_segment_dicts_to_ibd_summaries(ibd_segment_dicts)
    assert summaries == {
        frozenset([1, 2]): {
            "total_half": 73.667546 + 151.828402,
            "total_full": 7.544491000000001 + 59.876436,
            "num_half": 2,
            "max_seg_cm": 151.828402,
        }
    }


def test_transform_segment_lists_to_ibd_summaries_duplicates():
    ibd_segment_list = [
        [1, 2, "8", 80660500, 146364022, False, 73.667546],
        [2, 1, "8", 80660500, 146364022, False, 73.667546],
    ]

    with pytest.raises(ValueError):
        utils.transform_segment_lists_to_ibd_summaries(ibd_segment_list)


def test_transform_segment_dicts_to_ibd_summaries_duplicates():
    ibd_segment_dicts = [
        {
            "genotype_id_1": 1,
            "genotype_id_2": 2,
            "chromosome": "8",
            "start": 80660500,
            "end": 146364022,
            "is_full_ibd": False,
            "seg_cm": 73.667546,
        },
        {
            "genotype_id_1": 2,
            "genotype_id_2": 1,
            "chromosome": "8",
            "start": 80660500,
            "end": 146364022,
            "is_full_ibd": False,
            "seg_cm": 73.667546,
        },
    ]

    with pytest.raises(ValueError):
        utils.transform_segment_dicts_to_ibd_summaries(ibd_segment_dicts)


# =================== Test pedigree_object

def test_is_founder():
    up_dict = {1:('M',10,2,3), 2:('F',40,4,5), 4:('M',70,None,None), 5:('F',70,None)}
    ped_obj = PedigreeObject(up_dict)
    assert(ped_obj.is_founder(1) == False)
    assert(ped_obj.is_founder(2) == False)
    assert(ped_obj.is_founder(3) == True)
    assert(ped_obj.is_founder(4) == True)
    assert(ped_obj.is_founder(5) == True)
    up_dict = {1 : ('M',10)}
    ped_obj = PedigreeObject(up_dict)
    assert(ped_obj.is_founder(1) == True)
    up_dict = {1 : ('M',10,2)}
    ped_obj = PedigreeObject(up_dict)
    assert(ped_obj.is_founder(1) == False)


# =================== Test point prediction

def test_construct_point_prediction_group():
    with open(os.path.join(FIXTURES_DIR, "0_1_1.json")) as f:
        json_file = json.load(f)[0]

    bio_info = {
        i["genotype_id"]: {"age": i["age"], "sex": i["sex"]}
        for i in json_file["bio_info"]
    }

    segments_list = json_file["ibd_seg_list"]
    # Remove segments for individuals without bio_info
    filtered_segments = [
        s for s in segments_list if s[0] in bio_info and s[1] in bio_info
    ]

    segment_summaries = utils.transform_segment_lists_to_ibd_summaries(
        filtered_segments
    )
    prediction_group = point_predictor.construct_point_prediction_group(
        bio_info, segment_summaries
    )

    assert type(prediction_group) == point_predictor.PointPredictionGroup
    # Ensure that IBDSummary stats are being stored in the PointPredictionGroup
    assert all(
        [
            type(summary) == point_predictor.IBDSummary
            for summary in prediction_group.pairwise_segment_summaries.values()
        ]
    )
    assert len(prediction_group.total_half_len_list)  # Should have a non-zero length
    assert len(prediction_group.total_full_len_list)  # Should have a non-zero length
    assert len(prediction_group.half_seg_count_list)  # Should have a non-zero length
    assert len(prediction_group.max_seg_len_list)  # Should have a non-zero length


def test_fail_if_missing_summary_for_pair():
    segment_summaries = {
        frozenset([1, 2]): {
            "total_half": 1,
            "total_full": 1,
            "num_half": 1,
            "max_seg_cm": 1,
        },
        frozenset([1, 3]): {
            "total_half": 1,
            "total_full": 1,
            "num_half": 1,
            "max_seg_cm": 1,
        },
        # Missing summary for (2,3)
    }
    bio_info = {
        1: {"age": 30, "sex": "M"},
        2: {"age": 30, "sex": "M"},
        3: {"age": 30, "sex": "M"},
    }

    with pytest.raises(KeyError):
        point_predictor.construct_point_prediction_group(bio_info, segment_summaries)


# =================== Test node_dict_tools

def test_get_node_dict():
    up_dict = {
        1 : [None,None,-1,-2],
        2 : [None,None,-1,-2],
        3 : [None,None,-3,-4],
        4 : [None,None,-3,-4],
        5 : [None,None,-5,-6],
        -3 : [None,None,-7,-8],
        -5 : [None,None,-7,-8],
        -1 : [None,None,-9,-10],
        -9 : [None,None,-11,-12],
        -7 : [None,None,-11,-12],
    }

    po = PedigreeObject(up_dict)
    rel_set = {1}
    node_dict = get_node_dict(po, rel_set)
    assert(node_dict == {1 : {1 : 0}})

    po = PedigreeObject(up_dict)
    rel_set = {1,2}
    node_dict = get_node_dict(po, rel_set)
    assert(node_dict == {-1 : {1 : 1, 2 : 1}, -2 : {1 : 1 , 2 : 1}})

    rel_set = {1,2,3}
    node_dict = get_node_dict(po, rel_set)
    assert(node_dict == {-12: {-1: 2, 3: 3}, -11: {-1: 2, 3: 3}, -1: {1: 1, 2: 1}})

    rel_set = {1,2,3,4}
    node_dict = get_node_dict(po, rel_set)
    assert(node_dict == {-12: {-1: 2, -3: 2}, -11: {-1: 2, -3: 2}, -1: {1: 1, 2: 1}, -3 : {3 : 1, 4 : 1}})

    rel_set = {1,2,3,4,5}
    node_dict = get_node_dict(po, rel_set)
    assert(node_dict == {-12: {-1: 2, -7: 1}, -11: {-1: 2, -7: 1}, -7 : {-3 : 1, 5 : 2}, -1: {1: 1, 2: 1}, -3 : {3 : 1, 4 : 1}})


def test_get_node_dict_for_root():
    up_dict = {
        1 : [None,None,-1,-2],
        2 : [None,None,-1,-2],
        3 : [None,None,-3,-4],
        4 : [None,None,-3,-4],
        5 : [None,None,-5,-6],
        -3 : [None,None,-7,-8],
        -5 : [None,None,-7,-8],
        -1 : [None,None,-9,-10],
        -9 : [None,None,-11,-12],
        -7 : [None,None,-11,-12],
    }

    po = PedigreeObject(up_dict)
    root_id = 1
    node_dict = get_node_dict_for_root(root_id, po)
    assert(node_dict == {1 : {1 : 0}})

    root_id = -1
    node_dict = get_node_dict_for_root(root_id, po)
    assert(node_dict == {-1 : {1 : 1, 2 : 1}})

    root_id = -7
    node_dict = get_node_dict_for_root(root_id, po)
    assert(node_dict == {-7 : {-3 : 1, 5 : 2}, -3 : {3 : 1 , 4 : 1}})

    root_id = -11
    node_dict = get_node_dict_for_root(root_id, po)
    assert(node_dict == {-11: {-1: 2, -7: 1}, -7 : {-3 : 1, 5 : 2}, -1: {1: 1, 2: 1}, -3 : {3 : 1, 4 : 1}})

    root_id = -9
    node_dict = get_node_dict_for_root(root_id, po)
    assert(node_dict == {-9: {-1 : 1}, -1: {1: 1, 2: 1}})


def test_extract_down_node_dict():
    up_dict = {
        1 : [None,None,-1,-2],
        2 : [None,None,-1,-2],
        3 : [None,None,-3,-4],
        4 : [None,None,-3,-4],
        5 : [None,None,-5,-6],
        -3 : [None,None,-7,-8],
        -5 : [None,None,-7,-8],
        -1 : [None,None,-9,-10],
        -9 : [None,None,-11,-12],
        -7 : [None,None,-11,-12],
    }
    po = PedigreeObject(up_dict)

    rel_set = {1,2,3,4,5}
    node_dict = get_node_dict(po, rel_set)

    down_dict = extract_down_node_dict(1, node_dict)
    assert(down_dict == {})    

    try:
        down_dict = extract_down_node_dict(-9, node_dict)
    except MissingNodeException:
        pass

    down_dict = extract_down_node_dict(-1, node_dict)
    assert(down_dict == {-1: {1: 1, 2: 1}})

    down_dict = extract_down_node_dict(-7, node_dict)
    assert(down_dict == {-7 : {-3 : 1, 5 : 2}, -3 : {3 : 1, 4 : 1}})

    down_dict = extract_down_node_dict(-11, node_dict)
    assert(down_dict == {-11: {-1: 2, -7: 1}, -7 : {-3 : 1, 5 : 2}, -1: {1: 1, 2: 1}, -3 : {3 : 1, 4 : 1}})


def test_get_leaf_set():
    up_dict = {
        1 : [None,None,-1,-2],
        2 : [None,None,-1,-2],
        3 : [None,None,-3,-4],
        4 : [None,None,-3,-4],
        5 : [None,None,-5,-6],
        -3 : [None,None,-7,-8],
        -5 : [None,None,-7,-8],
        -1 : [None,None,-9,-10],
        -9 : [None,None,-11,-12],
        -7 : [None,None,-11,-12],
    }
    po = PedigreeObject(up_dict)

    rel_set = {1,2,3,4,5}
    node_dict = get_node_dict(po, rel_set)

    leaf_set = get_leaf_set(node_dict)
    assert(leaf_set == rel_set)


def test_get_root_to_desc_degrees():
    node_dict = {1 : {2 : 3 , 3 : 1} , 2 : {4 : 2, 5 : 2} , 3 : {6 : 2 , 7 : 1}}
    deg_dict = get_root_to_desc_degrees(1,node_dict)
    assert(deg_dict == {1:0, 2:3, 3:1, 4:5, 5:5, 6:3, 7:2})

    node_dict = {1 : {2 : 3 , 3 : 1} , 2 : {4 : 2, 5 : 2},  4 : {8 : 4}, 5 : {9 : 2, 10 : 10} , 3 : {6 : 2 , 7 : 1}}
    deg_dict = get_root_to_desc_degrees(1,node_dict)
    assert(deg_dict == {1:0, 2:3, 3:1, 4:5, 5:5, 8:9, 9:7, 10:15, 6:3, 7:2})

    node_dict = {1 : {1 : 0}}
    deg_dict = get_root_to_desc_degrees(1,node_dict)
    assert(deg_dict == {1 : 0})

    node_dict = {1 : {2 : 1, 3 : 2} , 2 : { 2 : 0 } , 3 : { 3 : 0 }}
    deg_dict = get_root_to_desc_degrees(1,node_dict)
    assert(deg_dict == {1:0, 2:1, 3:2})

    node_dict = {1 : {1 : 0}}
    deg_dict = get_root_to_desc_degrees(1,node_dict)
    assert(deg_dict == {1 : 0})

    node_dict = {}
    deg_dict = get_root_to_desc_degrees(1,node_dict)
    assert(deg_dict == {1 : 0})


def test_get_desc_deg_dict():
    node_dict = {1 : {2 : 3 , 3 : 1} , 2 : {4 : 1, 5 : 1} , 3 : {6 : 2 , 7 : 1}}
    indep_leaf_set = {4,5,6}
    anc_id = 1
    desc_deg_dict = get_desc_deg_dict(
        anc_id = anc_id,
        node_dict = node_dict,
    )
    desc_deg_dict = {i : desc_deg_dict[i] for i in indep_leaf_set}
    assert(desc_deg_dict == {4 : 4, 5 : 4, 6 : 3})
    indep_leaf_set = {4,5}
    desc_deg_dict = get_desc_deg_dict(
        anc_id = anc_id,
        node_dict = node_dict,
    )
    desc_deg_dict = {i : desc_deg_dict[i] for i in indep_leaf_set}
    assert(desc_deg_dict == {4 : 4, 5 : 4})
    indep_leaf_set = {2,7}
    desc_deg_dict = get_desc_deg_dict(
        anc_id = anc_id,
        node_dict = node_dict,
    )
    desc_deg_dict = {i : desc_deg_dict[i] for i in indep_leaf_set}
    assert(desc_deg_dict == {2 : 3, 7 : 2})

    node_dict = {1 : {1 : 0}}
    desc_deg_dict = get_desc_deg_dict(
        anc_id = 1,
        node_dict = node_dict,
    )
    assert(desc_deg_dict == {1 : 0})


def test_get_node_dict_founder_set():
    up_dict = {
        1 : [None,None,-1,-2],
        2 : [None,None,-1,-2],
        3 : [None,None,-3,-4],
        4 : [None,None,-3,-4],
        5 : [None,None,-5,-6],
        -3 : [None,None,-7,-8],
        -5 : [None,None,-7,-8],
        -1 : [None,None,-9,-10],
        -9 : [None,None,-11,-12],
        -7 : [None,None,-11,-12],
    }
    po = PedigreeObject(up_dict)

    rel_set = {1,2,3,4,5}
    node_dict = get_node_dict(po, rel_set)
    founder_set = get_node_dict_founder_set(node_dict)
    assert(founder_set ==  {-11, -12})

    rel_set = {1,2}
    node_dict = get_node_dict(po, rel_set)
    founder_set = get_node_dict_founder_set(node_dict)
    assert(founder_set ==  {-1, -2})    


def test_get_min_id():
    node_dict = {1 : {2 : 3 , 3 : 1} , 2 : {4 : 1, 5 : 1} , 3 : {6 : 2 , 7 : 1}}
    min_id = get_min_id(node_dict)
    assert(min_id == 1)
    node_dict = {1 : {2 : 3 , 3 : 1} , 2 : {4 : 1, 5 : 1} , 3 : {6 : 2 , -1 : 1}}
    min_id = get_min_id(node_dict)
    assert(min_id == -1)
    node_dict = {1 : {-5 : 3 , 3 : 1} , -5 : {4 : 1, 5 : 1} , 3 : {6 : 2 , -1 : 1}}
    min_id = get_min_id(node_dict)
    assert(min_id == -5)


def test_adjust_node_dict_to_common_anc():
    up_dict = {
        1 : [None,None,-1,-2],
        2 : [None,None,-1,-2],
        3 : [None,None,-3,-4],
        4 : [None,None,-3,-4],
        5 : [None,None,-5,-6],
        -3 : [None,None,-7,-8],
        -5 : [None,None,-7,-8],
        -1 : [None,None,-9,-10],
        -9 : [None,None,-11,-12],
        -7 : [None,None,-11,-12],
    }
    po = PedigreeObject(up_dict)
    node_dict = get_node_dict(
        ped_obj = po,
        rels = {3,4,5},
    )
    adj_node_dict = adjust_node_dict_to_common_anc(
        anc_id = -7,
        po = po,
        node_dict = node_dict,
    )
    assert(adj_node_dict == {-7 : {-3 : 1, 5 : 2}, -3 : {3 : 1, 4 : 1}})
    adj_node_dict = adjust_node_dict_to_common_anc(
        anc_id = -11,
        po = po,
        node_dict = node_dict,
    )
    assert(adj_node_dict == {-11 : {-7 : 1}, -7 : {-3 : 1, 5 : 2}, -3 : {3 : 1, 4 : 1}})  



# =================== Test analytical_likelihood_tools


def test_get_ibd_pattern_log_prob():
    node_dict = {1 : {2 : 3 , 3 : 1}}
    node = 1
    ibd_presence_absence_dict = {2 : False, 3 : False}
    result = get_ibd_pattern_log_prob(node, node_dict, ibd_presence_absence_dict)
    assert(result == (np.log(1),np.log(7/16)))

    node_dict = {1 : {2 : 3 , 3 : 1} , 3 : {4 : 1 , 5 : 1}}
    node = 1
    ibd_presence_absence_dict = {2 : False, 4 : True, 5 : False}
    result = get_ibd_pattern_log_prob(node, node_dict, ibd_presence_absence_dict)
    assert(result == (np.log(0),np.log(7/64)))
    # assign some random values to the internal nodes to make sure we get the same result
    ibd_presence_absence_dict = {1 : False, 3 : False, 2 : False, 4 : True, 5 : False}
    result = get_ibd_pattern_log_prob(node, node_dict, ibd_presence_absence_dict)
    assert(result == (np.log(0),np.log(7/64)))


def test_get_lambda_list():
    node_dict = {1 : {2 : 2, 3 : 1}, 2 : {4 : 2, 5 : 1}, 3 : {6 : 3, 7: 1}}
    num_common_ancs = 2
    lambda_list = get_lambda_list(
        node_dict = node_dict,
        indep_leaf_set1 = {4,5},
        indep_leaf_set2 = {6,7},
        root_id = 1,
        left_common_anc = 2,
        right_common_anc = 3,
    )
    diffs = [7, 5, 8, 6]
    lambda_set = {d/100 for d in diffs}
    assert(lambda_set == {*lambda_list})


def test_get_log_prob_ibd():
    node_dict = {1 : {2 : 1 , 3 : 1}}
    root_id = 1
    left_common_anc = 2
    right_common_anc = 3
    num_common_ancs = 1
    log_prob = get_log_prob_ibd(
        node_dict = node_dict,
        root_id = root_id,
        left_common_anc = left_common_anc,
        right_common_anc = right_common_anc,
        num_common_ancs = num_common_ancs,
        left_indep_leaf_set = {2},
        right_indep_leaf_set = {3},
    )
    assert log_prob == np.log(0.5)

    node_dict = {1 : {2 : 3 , 3 : 1} }
    root_id = 1
    left_common_anc = 2
    right_common_anc = 3
    num_common_ancs = 1
    log_prob = get_log_prob_ibd(
        node_dict = node_dict,
        root_id = root_id,
        left_common_anc = left_common_anc,
        right_common_anc = right_common_anc,
        num_common_ancs = num_common_ancs,
        left_indep_leaf_set = {2},
        right_indep_leaf_set = {3},
    )
    assert log_prob == np.log(1/8)

    node_dict = {1 : {2 : 3 , 3 : 1} , 2 : {4 : 1, 5 : 1} , 3 : {6 : 2 , 7 : 1}}
    root_id = 1
    left_common_anc = 2
    right_common_anc = 3
    num_common_ancs = 1
    log_prob = get_log_prob_ibd(
        node_dict = node_dict,
        root_id = root_id,
        left_common_anc = left_common_anc,
        right_common_anc = right_common_anc,
        num_common_ancs = num_common_ancs,
        left_indep_leaf_set = {4,5},
        right_indep_leaf_set = {6,7},
    )
    assert log_prob == np.log(15/256)

    node_dict = {1 : {2 : 2, 3 : 1}, 2 : {4 : 2, 5 : 1}, 3 : {6 : 3, 7: 1}}
    num_common_ancs = 2
    log_prob = get_log_prob_ibd(
        node_dict = node_dict,
        root_id = 1,
        left_common_anc = 2,
        right_common_anc = 3,
        num_common_ancs = num_common_ancs,
        left_indep_leaf_set = {4,5},
        right_indep_leaf_set = {6,7},
    )
    assert round(log_prob,4) == round(np.log(45/256),4)


def test_get_expected_seg_length_and_squared_length_for_leaf_subset():
    node_dict = {1 : {2 : 3 , 3 : 1} , 2 : {4 : 2, 5 : 2} , 3 : {6 : 2 , 7 : 1}}
    indep_leaf_set1 = {4,5}
    indep_leaf_set2 = {6,7}
    root_id = 1
    left_common_anc = 2
    right_common_anc = 3
    num_common_ancs = 1
    EL,EL2 = get_expected_seg_length_and_squared_length_for_leaf_subset(
        node_dict = node_dict,
        indep_leaf_set1 = indep_leaf_set1,
        indep_leaf_set2 = indep_leaf_set2,
        root_id = root_id,
        left_common_anc = left_common_anc,
        right_common_anc = right_common_anc,
    )
    assert(round(EL,2) == 27.97)
    assert(round(EL2,2) == 520.82)


def test_get_var_total_length_approx():
    node_dict = {1 : {2 : 3 , 3 : 1} , 2 : {4 : 2, 5 : 2} , 3 : {6 : 2 , 7 : 1}}
    indep_leaf_set1 = {4,5}
    indep_leaf_set2 = {6,7}
    root_id = 1
    left_common_anc = 2
    right_common_anc = 3
    num_common_ancs = 1
    log_prob_ibd = get_log_prob_ibd(
        node_dict = node_dict,
        root_id = root_id,
        left_common_anc = left_common_anc,
        right_common_anc = right_common_anc,
        num_common_ancs = num_common_ancs,
        left_indep_leaf_set = indep_leaf_set1,
        right_indep_leaf_set = indep_leaf_set2,
    )
    var_full,El,El2 = get_var_total_length_approx(
        node_dict = node_dict,
        indep_leaf_set1 = indep_leaf_set1,
        indep_leaf_set2 = indep_leaf_set2,
        root_id = root_id,
        left_common_anc = left_common_anc,
        right_common_anc = right_common_anc,
        num_common_ancs = num_common_ancs,
    )
    assert(round(var_full,2) == 1730.75)


def test_get_log_like_total_length_normal():
    L_tot = 10
    mean = 5
    var = 1
    log_like = get_log_like_total_length_normal(
        L_tot = L_tot,
        mean = mean,
        var = var,
    )
    assert(log_like == -13.418938533204672)


def test_get_background_test_pval_gamma():
    # Note: There are multiple ways to implement
    #       get_background_test_pval_gamma and a
    #       failed test is not necessarily indicative
    #       of a problem. This test is mostly useful 
    #       for testing whether the output has changed.
    L_tot = 10
    mean = 5
    var = 1
    pval = get_background_test_pval_gamma(
        L_tot = L_tot,
        mean = mean,
        var = var,
    )
    assert(pval == 3.454931382984584e-05)


def test_get_background_test_pval_normal():
    # Note: There are multiple ways to implement
    #       get_background_test_pval_normal and a
    #       failed test is not necessarily indicative
    #       of a problem. This test is mostly useful 
    #       for testing whether the output has changed.
    L_tot = 10
    mean = 5
    var = 1
    expected_count = 2
    pval = get_background_test_pval_normal(
        L_tot = L_tot,
        mean = mean,
        var = var,
        expected_count = expected_count,
    )
    assert(pval == 1.285513908114333e-06)


# ==================== Test ibd_tools.py


def test_get_related_sets():
    data_path = os.path.join(FIXTURES_DIR, '4gens_2offspring_0.1probhalf.json') #'test_validated_peds2.json')
    ped_data = json.loads(open(data_path).read())
    ibd_seg_list = ped_data['ibd_seg_list']
    rels1,rels2 = get_related_sets({1,4,67,68},{215,216},ibd_seg_list)
    assert(rels1 == {1,4})
    assert(rels2 == {215,216})


# ==================== Test connect_pedigree_tools.py


# def test_get_connecting_anc_pair_deg_Ltot_and_log_like()
#   No test for this. We test subroutines instead.


# def test_get_open_ancestor_set()
#   No test for this. It is a subroutine of get_open_ancestor_set_for_leaves()


# def test_check_pred_deg_likes():
#   This function is somewhat complicatted and is not tested explicitly    


def test_infer_degree_generalized_druid():
    # start with simple test.
    # These estimates were generated by running infer_degree_generalized_druid()
    # so they represent the state of the function aat the time of writing.
    # Failure on this test does not necessarily indicate a problem, only a change.
    node_dict1 = {1 : {2 : 3, 3 : 1} , 2 : {4 : 1, 5 : 1} , 3 : {6 : 2, 7 : 1}}
    node_dict2 = {8 : {9 : 2, 10 : 3}}
    leaf_set1 = {2,6,7}
    leaf_set2 = {9,10}
    num_common_ancs = 2
    L = 20
    est_deg = infer_degree_generalized_druid(leaf_set1, leaf_set2, node_dict1, node_dict2, L)
    assert(est_deg == 6)
    L = 100
    est_deg = infer_degree_generalized_druid(leaf_set1, leaf_set2, node_dict1, node_dict2, L)
    assert(est_deg == 3)
    L = 200
    est_deg = infer_degree_generalized_druid(leaf_set1, leaf_set2, node_dict1, node_dict2, L)
    assert(est_deg == 2)

    # run a more complicated test
    filename = 'nreps=1_pedtype=deterministic_lambda_numgens1=1through10_numgens2=1_ncommonancs=2_split_gen_vec=[1,1].json'
    file_path = os.path.join(FIXTURES_DIR, filename)
    fixture_data = json.loads(open(file_path).read())

    true_deg_list = []
    est_deg_list = []
    deg_diff_list = []
    for num_gens, rep_data in fixture_data.items():
        # load data
        true_up_dict = {int(uid) : info for uid, info in rep_data['true_up_dict'].items()}
        ibd_seg_list = rep_data['ibd_seg_list']
        bio_info = rep_data['bio_info']

        # initialize pedigree object
        po = PedigreeObject(true_up_dict)

        # find leaf sets
        kid1,kid2 = po.down_pedigree_dict[1][2:]
        if po.is_leaf(kid1):
            leaf_set1 = {kid1}
        else:
            leaf_set1 = {uid for uid in po.rel_dict[kid1]['desc'] if po.is_leaf(uid)}
        if po.is_leaf(kid2):
            leaf_set2 = {kid2}
        else:
            leaf_set2 = {uid for uid in po.rel_dict[kid2]['desc'] if po.is_leaf(uid)}
        leaf_set = leaf_set1 | leaf_set2

        # find shared IBD
        chrom_ibd_segs_dict = get_ibd_segs_between_sets(
                                    leaves1 = leaf_set1,
                                    leaves2 = leaf_set2,
                                    ibd_seg_list = ibd_seg_list,
                                )

        # merge shared IBD
        merged_chrom_ibd_segs_dict = merge_ibd_segs(chrom_ibd_segs_dict)
        map_interpolator = get_map_interpolator()
        seg_len_list = get_segment_length_list(merged_chrom_ibd_segs_dict)
        L_merged_tot = sum(seg_len_list)

        # if there is no IBD, we can't infer anything, so continue
        if L_merged_tot == 0:
            continue

        # get number of common ancestors
        root_id = 1
        common_anc_dict1 = po.get_common_ancestor_dict(list(leaf_set1),get_mrcas=True)
        common_anc_dict2 = po.get_common_ancestor_dict(list(leaf_set2),get_mrcas=True)
        for ca1 in common_anc_dict1.keys():
            if 1 in po.rel_dict[ca1]['anc']:
                break
        for ca2 in common_anc_dict2.keys():
            if 1 in po.rel_dict[ca2]['anc']:
                break
        num_common_ancs = po.rels[ca1][ca2][2]
        node_dict = get_node_dict(po,list(leaf_set))

        # get node dicts in preparation for inferring DRUID degree
        node_dict1 = extract_down_node_dict(ca1,node_dict)
        node_dict2 = extract_down_node_dict(ca2,node_dict)

        # infer the DRUID degree
        est_deg_druid = infer_degree_generalized_druid(
            leaf_set1 = leaf_set1,
            leaf_set2 = leaf_set2,
            node_dict1 = node_dict1,
            node_dict2 = node_dict2,
            L_merged_tot = L_merged_tot,
        )

        # get the true degree
        true_deg = node_dict[root_id][ca1] + node_dict[root_id][ca2] - num_common_ancs + 1

        # store true, est, and diff of degrees
        true_deg_list.append(int(true_deg))
        est_deg_list.append(int(est_deg_druid))
        deg_diff_list.append(abs(true_deg - est_deg_druid))

        # test accuracy of inference. Allow less accurate estimates for large degrees
        # NOTE: failure here does not necessarily indicate a problem, it just indicates
        #       that the estimate has changed and potentially flags problems.
        if true_deg <= 4:
            assert(true_deg == est_deg_druid)
        elif true_deg <= 6:
            assert(abs(true_deg - est_deg_druid) <= 1)
        else:
            assert(abs(true_deg - est_deg_druid) <= 2)


def test_get_open_ancestor_set_for_leaves():
    data_path = os.path.join(FIXTURES_DIR, 'test_tree_combining.json')
    data = json.loads(open(data_path).read())

    leaf_set1 = {*data['leaf_set1']}
    leaf_set2 = {*data['leaf_set2']}
    up_dict1 = data['up_dict1']
    up_dict2 = data['up_dict2']
    ibd_seg_list = data['ibd_seg_list']

    up_dict1 = {int(k) : v for k,v in up_dict1.items()} # un-jsonify keys
    up_dict2 = {int(k) : v for k,v in up_dict2.items()}

    po1 = PedigreeObject(up_dict1)
    po2 = PedigreeObject(up_dict2)

    open_set1, half_open_set1, half_open_set_overlap1 = get_open_ancestor_set_for_leaves(
        po = po1,
        leaf_set = {1,4,63,64},
        other_leaf_set = leaf_set2,
        ibd_seg_list = ibd_seg_list,
        threshold = 0.05,
        use_overlap = True,
    )
    assert(open_set1 == {-1,-3,1000})
    assert(half_open_set1 == set())
    assert(half_open_set_overlap1 == set())

    open_set2, half_open_set2, half_open_set_overlap2 = get_open_ancestor_set_for_leaves(
        po = po2,
        leaf_set = {215,216},
        other_leaf_set = leaf_set2,
        ibd_seg_list = ibd_seg_list,
        threshold = 0.05,
        use_overlap = True,
    )
    assert(open_set2 == {214})
    assert(half_open_set2 == set())
    assert(half_open_set_overlap2 == set())


def test_remove_symmetric_ancestors():
    up_dict = {
        1 : [None,None,-1,-2],
        2 : [None,None,-1,-2],
        3 : [None,None,-3,-4],
        4 : [None,None,-3,-4],
        5 : [None,None,-5,-6],
        -3 : [None,None,-7,-8],
        -5 : [None,None,-7,-8],
        -1 : [None,None,-9,-10],
        -9 : [None,None,-11,-12],
        -7 : [None,None,-11,-12],
        -6 : [None,None,-13,-14],
    }
    po = PedigreeObject(up_dict)
    potential_open_ancestor_set = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14}
    open_ancestor_set = remove_symmetric_ancestors(
        open_set = potential_open_ancestor_set,
        po = po,
    )
    # check that we have removed either -12 and -14
    assert(open_ancestor_set == {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -13})


def test_get_connecting_founders_degs_and_log_likes():
    # do a simple test
    data_path = os.path.join(FIXTURES_DIR, 'test_get_connecting_founders_degs_and_log_likes.json')
    data = json.loads(open(data_path).read())

    leaf_set1 = {*data['leaf_set1']}
    leaf_set2 = {*data['leaf_set2']}
    up_dict1 = data['up_dict1']
    up_dict2 = data['up_dict2']
    ibd_seg_list = data['ibd_seg_list']

    up_dict1 = {int(k) : v for k,v in up_dict1.items()} # un-jsonify keys
    up_dict2 = {int(k) : v for k,v in up_dict2.items()}

    po1 = PedigreeObject(up_dict1)
    po2 = PedigreeObject(up_dict2)

    anc_deg_log_like_list = get_connecting_founders_degs_and_log_likes(
        po1 = po1,
        po2 = po2,
        gt_set1 = {1,4,63,64},
        gt_set2 = {215,216},
        ibd_seg_list = ibd_seg_list,
        require_descendant_match = True,
    )

    possible_connections = {(-3, -18, 3), (-8, -18, 3), (-1, -18, 4)}
    for est_ca1, est_ca2, est_deg, log_like in anc_deg_log_like_list:
        assert((est_ca1,est_ca2,est_deg) in possible_connections)

    # do a more complicated test
    filename = 'nreps=5_pedtype=deterministic_unrel_lambda_numgens1=2_numgens2=2_ncommonancs=2_split_gen_vec=[0,0,1,1].json'
    data_path = os.path.join(FIXTURES_DIR, filename)
    ped_data_list = json.loads(open(data_path).read())

    ped_data = ped_data_list[0]

    for ped_data in ped_data_list:

        true_up_dict = distributions.unstringify_keys(ped_data['true_up_dict'])
        ibd_seg_list = ped_data['ibd_seg_list']
        bio_info = ped_data['bio_info']

        sex_dict = {info['genotype_id'] : info['sex'] for info in bio_info}
        age_dict = {info['genotype_id'] : info['age'] for info in bio_info}

        true_ped_obj = PedigreeObject(true_up_dict)

        all_ids = {*sex_dict}

        profile_information = {info['genotype_id'] : {'age' : info['age'], 'sex' : info['sex']} for info in bio_info if info['genotype_id'] in all_ids}
        ibd_stat_dict = utils.transform_segment_lists_to_ibd_summaries(ibd_seg_list)

        distns = distributions.load_distributions()
        point_pred_group = point_predictor.construct_point_prediction_group(profile_information, ibd_stat_dict)
        pw_rels,pw_log_likes = point_predictor.point_predictions(
            point_prediction_group = point_pred_group,
            distribution_models = distns,
        )

        root_id1 = 3  # left child directly below root node of lambda pedigree
        root_id2 = 4  # right child directly below root node of lambda pedigree

        all_leaf_set = {uid for uid in true_ped_obj.up_pedigree_dict if true_ped_obj.is_leaf(uid)}  # all genotyped leaf ids
        gt_set1 = {uid for uid in true_ped_obj.rel_dict[3]['desc'] if true_ped_obj.is_leaf(uid)}  # gt ids in left clade related to gt_set2
        gt_set2 = {uid for uid in true_ped_obj.rel_dict[4]['desc'] if true_ped_obj.is_leaf(uid)}  # gt ids in right clade related to gt_set1
        outlier_set = all_leaf_set - gt_set1 - gt_set2  # set of gentoyped IDs related to gt_set1, but not gt_set2

        # make two pedigrees to join
        po1 = copy.deepcopy(true_ped_obj)
        po2 = copy.deepcopy(true_ped_obj)

        po1.keep_nodes(keep_gt_node_set = gt_set1 | outlier_set, include_parents=False)
        po2.keep_nodes(keep_gt_node_set = gt_set2, include_parents=False)

        # replace genotyped ancestors with ungenotyped        
        po1_non_leaf_set = [uid for uid in po1.down_pedigree_dict.keys()]
        for uid in sorted(po1_non_leaf_set):
            old_sex =po1.down_pedigree_dict[uid][0]
            old_age =po1.down_pedigree_dict[uid][1]
            new_id = po1.get_new_ind()
            po1.replace_individual(uid, new_id, old_sex, old_age)

        po2_non_leaf_set = [uid for uid in po2.down_pedigree_dict.keys()]
        for uid in sorted(po2_non_leaf_set):
            old_sex =po2.down_pedigree_dict[uid][0]
            old_age =po2.down_pedigree_dict[uid][1]
            new_id = po2.get_new_ind()
            po2.replace_individual(uid, new_id, old_sex, old_age)

        # get ancestors of just gt_set1 (these are the correct ancestors we should join when combining the pedigrees)
        common_anc_set1 = {*po1.get_common_ancestor_dict(list(gt_set1), get_mrcas=True)}
        # get the ancestors of both gt_set1 and outlier_set. It is incorrect to join the pedigrees through these ancestors.
        outlier_common_anc_set = {*po1.get_common_ancestor_dict(list(gt_set1 | outlier_set), get_mrcas=True)}

        # get any further ancestors of common_anc_set1.
        common_anc_pid_set1 = set()
        for ca_id in common_anc_set1:
            pid_set = {*po1.get_ancestor_dict(ca_id)}
            common_anc_pid_set1 |= pid_set
        common_anc_set1 |= common_anc_pid_set1

        # get any further ancestors of outlier_common_anc_set
        outlier_common_anc_pid_set = set()
        for ca_id in outlier_common_anc_set:
            pid_set = {*po1.get_ancestor_dict(ca_id)}
            outlier_common_anc_pid_set |= pid_set
        outlier_common_anc_set |= outlier_common_anc_pid_set

        # it is correct to join the pedigrees through any ancestor of common_anc_set1 who is not
        # also an ancestor of outlier_common_anc_set
        correct_join_ancestors = common_anc_set1 - outlier_common_anc_set

        # make sure no ungenotyped index of po2 overlaps with any ungenotyped index of po1
        min_ind1 = po1.min_parent_ind
        min_ind2 = po2.min_parent_ind
        min_ind = min(min_ind1,min_ind2)
        po2.update_ungenotyped_inds(min_ind-1)

        anc_deg_log_like_list = get_connecting_founders_degs_and_log_likes(
            po1 = po1,
            po2 = po2,
            gt_set1 = gt_set1,
            gt_set2 = gt_set2,
            ibd_seg_list = ibd_seg_list,
        )

        max_like_arrangement = anc_deg_log_like_list[0]
        ca1,ca2 = max_like_arrangement[:2]
        assert(ca1 in correct_join_ancestors)


def test_get_deg1_deg2():

    data_path = os.path.join(FIXTURES_DIR, 'test_get_connecting_founders_degs_and_log_likes.json')
    data = json.loads(open(data_path).read())

    leaf_set1 = {*data['leaf_set1']}
    leaf_set2 = {*data['leaf_set2']}
    up_dict1 = data['up_dict1']
    up_dict2 = data['up_dict2']
    ibd_seg_list = data['ibd_seg_list']

    up_dict1 = {int(k) : v for k,v in up_dict1.items()} # un-jsonify keys
    up_dict2 = {int(k) : v for k,v in up_dict2.items()}

    po1 = PedigreeObject(up_dict1)
    po2 = PedigreeObject(up_dict2)

    ca1 = -3
    ca2 = -18
    deg = 3

    node_dict1 = get_node_dict_for_root(ca1, po1)
    node_dict2 = get_node_dict_for_root(ca2, po2)
    deg1, deg2, log_like = get_deg1_deg2(
        deg = deg,
        anc_id1 =ca1,
        anc_id2 = ca2,
        po1 = po1,
        po2 = po2,
        num_common_ancs = 2,
    )
    assert((deg1,deg2) == (1, 3))


def test_find_open_partner_and_update_po():
    data_path = os.path.join(FIXTURES_DIR, '4gens_2offspring_0.1probhalf.json') #'test_validated_peds2.json')
    ped_data = json.loads(open(data_path).read())
    true_up_dict = ped_data['true_up_dict']

    new_up_dict = dict()
    for uid,info in true_up_dict.items():
        new_id = int(uid)
        new_up_dict[new_id] = info

    ped_obj = PedigreeObject(new_up_dict)
    gt_set = {151, 152, 215, 219, 220, 216}
    ped_obj.keep_nodes(gt_set,include_parents=False)

    partner_id = find_open_partner_and_update_po(
        node_id = 214, 
        #gt_desc_set = {215,216}, 
        po = ped_obj,
    )
    assert(partner_id == None)

    partner_id = find_open_partner_and_update_po(
        node_id = 151, 
        #gt_desc_set = {215,216}, 
        po = ped_obj,
    )
    assert(partner_id == 214)

    partner_id = find_open_partner_and_update_po(
        node_id = 150, 
        #gt_desc_set = gt_set, 
        po = ped_obj,
    )
    assert(partner_id == 87)

    ped_obj.remove_node(87)
    partner_id = find_open_partner_and_update_po(
        node_id = 150, 
        #gt_desc_set = gt_set, 
        po = ped_obj,
    )
    assert(partner_id == None)


def test_get_best_desc_id_set():
    up_dict = {
        1 : [None,None,-1,-2],
        2 : [None,None,-1,-2],
        3 : [None,None,-3,-4],
        4 : [None,None,-4,-5],
        -1 : [None,None,-6,-7],
        -3 : [None,None,-6,-7],
    }
    po = PedigreeObject(up_dict)

    gt_id_set = {1,2,3,4}
    other_id_set = {5}

    ibd_seg_list = [
        [1, 5, '3', 37706267, 66391837, False, 28.025470733642578],
        [2, 5, '3', 37706267, 66391837, False, 28.025470733642578],
        [3, 5, '3', 37706267, 66391837, False, 28.025470733642578],
        [4, 5, '17', 55701470, 80890638, False, 46.832611083984375],
    ]

    best_desc_id_set = get_best_desc_id_set(
        gt_id_set = gt_id_set,
        other_gt_id_set = other_id_set,
        po = po,
        ibd_seg_list = ibd_seg_list,
    )

    assert(best_desc_id_set == {3,4})

    gt_id_set = {1,4}
    other_id_set = {5}

    ibd_seg_list = [
        [1, 5, '3', 37706267, 66391837, False, 28.025470733642578],
        [4, 5, '17', 55701470, 80890638, False, 46.832611083984375],
    ]

    best_desc_id_set = get_best_desc_id_set(
        gt_id_set = gt_id_set,
        other_gt_id_set = other_id_set,
        po = po,
        ibd_seg_list = ibd_seg_list,
    )

    assert(best_desc_id_set == {4})


def test_combine_pedigrees():

    data_path = os.path.join(FIXTURES_DIR, 'test_get_connecting_founders_degs_and_log_likes.json')
    data = json.loads(open(data_path).read())

    leaf_set1 = {*data['leaf_set1']}
    leaf_set2 = {*data['leaf_set2']}
    up_dict1 = data['up_dict1']
    up_dict2 = data['up_dict2']
    ibd_seg_list = data['ibd_seg_list']

    up_dict1 = {int(k) : v for k,v in up_dict1.items()} # un-jsonify keys
    up_dict2 = {int(k) : v for k,v in up_dict2.items()}

    po1 = PedigreeObject(up_dict1)
    po2 = PedigreeObject(up_dict2)

    new_ped_obj_list = combine_pedigrees(
        po1 = po1,
        po2 = po2,
        ibd_seg_list = ibd_seg_list,
    )
    assert(new_ped_obj_list[0].rels[1][215] == (4,4,2))


def test_find_closest_pedigrees():
    index_to_gtid_set = {0 : {1,2}, 2 : {3, 4}, 3 : {5}}
    ibd_seg_list = [
        [1, 3, '3', 37706267, 66391837, False, 28.025470733642578],
        [2, 4, '3', 37706267, 66391837, False, 28.025470733642578],
        [1, 5, '3', 37706267, 66391837, False, 28.025470733642578],
        [3, 5, '3', 37706267, 66391837, False, 28.025470733642578],
        [4, 5, '17', 55701470, 80890638, False, 46.832611083984375],
    ]
    closest_ped_idxs = find_closest_pedigrees(
        index_to_gtid_set = index_to_gtid_set,
        ibd_seg_list = ibd_seg_list,
    )
    assert({*closest_ped_idxs} == {2,3})


def test_drop_background_ibd():
    up_dict1 = {
        1 : [None, None, -1, -2],
        2 : [None, None, -1, -2],
        3 : [None, None, -3, -4],
        4 : [None, None, -3, -4],
        -1 : [None, None, -5, -6],
        -3 : [None, None, -5, -7],
    }
    po1 = PedigreeObject(up_dict1)

    up_dict2 = {
        5 : [None, None]
    }
    po2 = PedigreeObject(up_dict2)

    ibd_seg_list = [
        [1, 5, '3', 37706267, 66391837, False, 28.025470733642578],
        [3, 5, '3', 37706267, 66391837, False, 28.025470733642578],
        [3, 5, '17', 12456159, 81041077, False, 96.75493621826172],
        [4, 5, '17', 55701470, 80890638, False, 46.832611083984375],
    ]

    new_ca1 = drop_background_ibd(
        ca1 = -5,
        ca2 = 5,
        po1 = po1,
        po2 = po2,
        ibd_seg_list = ibd_seg_list,
        alpha = 0.01,
    )

    assert(new_ca1 == -3)


# ======================= Test pedigree_object.py

def test_extend_up_and_extend_down():

    data_path = os.path.join(FIXTURES_DIR, 'test_get_connecting_founders_degs_and_log_likes.json')
    data = json.loads(open(data_path).read())

    up_dict1 = data['up_dict1']
    up_dict1 = {int(k) : v for k,v in up_dict1.items()} # un-jsonify keys
    po1 = PedigreeObject(up_dict1)

    ca1 = -1
    po1,root_id, prev_id = extend_up(ca1,2,po1)
    assert(po1.rels[ca1][root_id] == (2,0,1))
    assert(po1.rels[ca1][prev_id] == (1,0,1))

    partner_id = po1.add_parent_for_child(child_id = prev_id)
    po1,last_id, prev_id = extend_down(root_id,2,po1,partner_id=partner_id)
    assert(po1.rels[ca1][last_id] == (2,2,2))

    partner_id = None
    po1,last_id, prev_id = extend_down(root_id,1,po1,partner_id=partner_id)
    assert(po1.rels[ca1][last_id] == (2,1,1))


def test_connect_pedigrees_through_founders():

    data_path = os.path.join(FIXTURES_DIR, 'test_get_connecting_founders_degs_and_log_likes.json')
    data = json.loads(open(data_path).read())

    leaf_set1 = {*data['leaf_set1']}
    leaf_set2 = {*data['leaf_set2']}
    up_dict1 = data['up_dict1']
    up_dict2 = data['up_dict2']
    ibd_seg_list = data['ibd_seg_list']

    up_dict1 = {int(k) : v for k,v in up_dict1.items()} # un-jsonify keys
    up_dict2 = {int(k) : v for k,v in up_dict2.items()}

    po1 = PedigreeObject(up_dict1)
    po2 = PedigreeObject(up_dict2)

    ca1 = -3
    ca2 = -18
    
    deg1 = 1
    deg2 = 3
    po1_copy = copy.deepcopy(po1)
    po2_copy = copy.deepcopy(po2)
    new_ped_obj = connect_pedigrees_through_founders(
        anc_id1 = ca1,
        anc_id2 = ca2,
        po1 = po1_copy,
        po2 = po2_copy,
        deg1 = deg1,
        deg2 = deg2,
        partner_id1 = None,
        partner_id2 = None,
    )
    assert(new_ped_obj.rels[1][215] == (4,4,2))

    deg1 = 0
    deg2 = 3
    po1_copy = copy.deepcopy(po1)
    po2_copy = copy.deepcopy(po2)
    new_ped_obj = connect_pedigrees_through_founders(
        anc_id1 = ca1,
        anc_id2 = ca2,
        po1 = po1_copy,
        po2 = po2_copy,
        deg1 = deg1,
        deg2 = deg2,
        partner_id1 = None,
        partner_id2 = None,
    )
    assert(new_ped_obj.rels[1][215] == (3,4,1))

    ca1 = -3
    ca2 = -18
    deg1 = 0
    deg2 = 2
    partner_id1 = -8
    partner_id2 = None
    po1_copy = copy.deepcopy(po1)
    po2_copy = copy.deepcopy(po2)
    new_ped_obj = connect_pedigrees_through_founders(
        anc_id1 = ca1,
        anc_id2 = ca2,
        po1 = po1_copy,
        po2 = po2_copy,
        deg1 = deg1,
        deg2 = deg2,
        partner_id1 = partner_id1,
        partner_id2 = partner_id2,
    )
    assert(new_ped_obj.rels[1][215] == (3,3,2))

    ca1 = -1
    ca2 = -14
    partner_id1 = None
    partner_id2 = -16
    deg1 = 2
    deg2 = 0
    po1_copy = copy.deepcopy(po1)
    po2_copy = copy.deepcopy(po2)
    new_ped_obj = connect_pedigrees_through_founders(
        anc_id1 = ca1,
        anc_id2 = ca2,
        po1 = po1_copy,
        po2 = po2_copy,
        deg1 = deg1,
        deg2 = deg2,
        partner_id1 = partner_id1,
        partner_id2 = partner_id2,
    )
    assert(new_ped_obj.rels[1][215] == (4,2,2))

    ca1 = -1
    ca2 = -14
    partner_id1 = -2
    partner_id2 = None
    deg1 = 0
    deg2 = 1
    po1_copy = copy.deepcopy(po1)
    po2_copy = copy.deepcopy(po2)
    new_ped_obj = connect_pedigrees_through_founders(
        anc_id1 = ca1,
        anc_id2 = ca2,
        po1 = po1_copy,
        po2 = po2_copy,
        deg1 = deg1,
        deg2 = deg2,
        partner_id1 = partner_id1,
        partner_id2 = partner_id2,
    )
    assert(new_ped_obj.rels[1][215] == (2,3,2))

    ca1 = -1
    ca2 = -14
    partner_id1 = None
    partner_id2 = None
    deg1 = 0
    deg2 = 1
    po1_copy = copy.deepcopy(po1)
    po2_copy = copy.deepcopy(po2)
    new_ped_obj = connect_pedigrees_through_founders(
        anc_id1 = ca1,
        anc_id2 = ca2,
        po1 = po1_copy,
        po2 = po2_copy,
        deg1 = deg1,
        deg2 = deg2,
        partner_id1 = partner_id1,
        partner_id2 = partner_id2,
    )
    assert(new_ped_obj.rels[1][215] == (2,3,1))

    ca1 = -1
    ca2 = -14
    partner_id1 = -2
    partner_id2 = -16
    deg1 = 0
    deg2 = 0
    po1_copy = copy.deepcopy(po1)
    po2_copy = copy.deepcopy(po2)
    new_ped_obj = connect_pedigrees_through_founders(
        anc_id1 = ca1,
        anc_id2 = ca2,
        po1 = po1_copy,
        po2 = po2_copy,
        deg1 = deg1,
        deg2 = deg2,
        partner_id1 = partner_id1,
        partner_id2 = partner_id2,
    )
    assert(new_ped_obj.rels[1][215] == (2,2,2))



# ======================= Test build_pedigree.py



#@pytest.mark.skip
def test_infer_local_pedigrees():
    data_path = os.path.join(FIXTURES_DIR, '4gens_2offspring_0.1probhalf.json')
    ped_data = json.loads(open(data_path).read())

    focal_id = ped_data['focal_id']
    true_up_dict = ped_data['true_up_dict']
    ibd_seg_list = ped_data['ibd_seg_list']
    bio_info = ped_data['bio_info']
    sex_dict = ped_data['sex_dict']
    age_dict = ped_data['age_dict']

    new_up_dict = dict()
    for uid,info in true_up_dict.items():
        new_id = int(uid)
        new_up_dict[new_id] = info
    true_up_dict = new_up_dict

    true_ped_obj = PedigreeObject(true_up_dict)

    all_ids = {uid for uid in true_ped_obj.up_pedigree_dict.keys() if uid > 0} | {uid for uid in true_ped_obj.down_pedigree_dict.keys() if uid > 0}
    del_set = {uid for uid in all_ids if true_ped_obj.rels[focal_id].get(uid,(INF,INF,None))[2] is None}
    all_ids -= del_set
    
    profile_information = {info['genotype_id'] : {'age' : info['age'], 'sex' : info['sex']} for info in bio_info if info['genotype_id'] in all_ids}
    all_ids = set(profile_information.keys())
    ibd_stat_dict = utils.transform_segment_lists_to_ibd_summaries(ibd_seg_list)
    new_ibd_stat_dict = copy.deepcopy(ibd_stat_dict)
    for key,info in ibd_stat_dict.items():
        if len(key & all_ids) < 2:
            del new_ibd_stat_dict[key]
    ibd_stat_dict = new_ibd_stat_dict

    distns = distributions.load_distributions()
    point_pred_group = point_predictor.construct_point_prediction_group(profile_information, ibd_stat_dict)
    pw_rels,pw_log_likes = point_predictor.point_predictions(
        point_prediction_group = point_pred_group,
        distribution_models = distns,
    )

    sex_dict = {uid : true_up_dict[uid][0] for uid in all_ids}
    age_dict = {uid : true_up_dict[uid][1] for uid in all_ids}

    build_frac = 0.25
    sample_size = int(np.ceil(build_frac * len(all_ids)))

    num_reps = 20
    max_frac_incorrect = 0
    random.seed(1982)
    for rep in range(num_reps):

        sample_id_set = set(random.sample([*all_ids], sample_size)) | {focal_id}
        new_sex_dict = {uid : sex_dict[uid] for uid in sample_id_set}
        new_age_dict = {uid : age_dict[uid] for uid in sample_id_set}
        new_pw_rels = dict()
        for id1,info in pw_rels.items():
            for id2,deg in info.items():
                if len({id1,id2} & sample_id_set) == 2:
                    if id1 not in new_pw_rels:
                        new_pw_rels[id1] = dict()
                    new_pw_rels[id1][id2] = deg
        new_pw_log_likes = dict()
        for id1,info in pw_log_likes.items():
            for id2,log_like_dict in info.items():
                if len({id1,id2} & sample_id_set) == 2:
                    if id1 not in new_pw_log_likes:
                        new_pw_log_likes[id1] = dict()
                    new_pw_log_likes[id1][id2] = log_like_dict

        result = infer_local_pedigrees(
            focal_id = focal_id,
            sex_dict = new_sex_dict,
            age_dict = new_age_dict,
            pw_rels = new_pw_rels,
            pw_log_likes = new_pw_log_likes,
            ibd_stat_dict = ibd_stat_dict,
            max_radius = INF,
            max_add_degree = 4,
            min_rel_append_types = 1,
            max_rel_append_types = 2,
            ped_save_log_like_delta_fraction = 0.333,
            ped_save_log_like_abs_fraction = 0.1,
            disallow_distant_half_rels = False,
            use_age_info = True,
        )
        index_to_gtid_set, index_to_ped_obj_list, gtid_to_ped_obj_index, traces_dict = result

        mismatch_set = set()
        total_pairs = 0
        for index,id_set in index_to_gtid_set.items():
            ped_obj_list = index_to_ped_obj_list[index]
            ped_obj = ped_obj_list[0]
            for id1,id2 in combinations(id_set, r=2):
                est_deg = ped_obj.rels[id1][id2]
                true_deg = true_ped_obj.rels[id1][id2]
                est_abs_deg = est_deg[0] + est_deg[1] - est_deg[2] + 1 if est_deg[2] is not None else INF
                true_abs_deg = true_deg[0] + true_deg[1] - true_deg[2] + 1 if true_deg[2] is not None else INF
                if true_abs_deg < 5 and est_deg != true_deg:
                    mismatch_set.add((id1,id2,true_deg,est_deg))
                total_pairs += 1

        frac_incorrect = len(mismatch_set) / total_pairs
        max_frac_incorrect = max(frac_incorrect, max_frac_incorrect)

    assert(max_frac_incorrect < 0.05)
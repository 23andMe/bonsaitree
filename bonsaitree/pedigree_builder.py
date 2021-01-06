from six import iteritems
import os
from collections import defaultdict
import copy
import logging
import numbers
import numpy as np
import pdb
import warnings

from itertools import combinations

from .copytools import deepcopy
from .pedigree_object import PedigreeObject
from .exceptions import AgeInconsistentException, UnlikelyRelationshipException

from .distributions import AUTO_GENOME_LENGTH, MIN_PARENT_CHILD_AGE_DIFF, SMALL_LOG_PROB

from .point_predictor import is_twin_pair


class PedigreeBuilder(object):
    """
    Do a branch and bound approach with multiple pedigree objects on each step.
    For each ped_obj existing on step i, take the new_id and add it to the pedigree
    to the close individual in three possible ways, corresponding to the three most
    likely pairwise relationships between closest_rel_id and new_id. Compute the total
    pedigree likelihood for all three pedigrees, keep the highest-likelihood one (or
    highest tied likelihood ones). You now have a set of pedigrees, one or more for
    each pedigree that existed on step i-1. Throw out all but the highest likelihood
    pedigrees. Then move on to step i+1 and repeat.
    """

    def __init__(self, pairwise_relationships,
                       pairwise_log_likelihoods,
                       ibd_stat_dict,
                       sex_dict,
                       age_dict,
                       use_age_info=True,
                       min_rel_append_types=1,
                       max_rel_append_types=float('inf'),
                       plot_trace_path=None):

        self.ped_obj_list = [] # Lists pedigree objects and their likelihoods.
        self.ped_log_like_list = [] # same order as self.ped_obj_list. stores likelihoods of the pedigrees in self.ped_obj_list
        self.pw_rels = pairwise_relationships # Point estimates of pairwise relationships. Has the form pw_rels[ind_1][ind_2] = [est_up_meioses,est_down_meioses,est_num_common_ancs]
        self.pw_log_likes = pairwise_log_likelihoods # Of the form pw_log_likes[ind1][ind2][(up,down,num_ancs)] = log_like_val
        self.ibd_stat_dict = ibd_stat_dict # Dictionary of the form ibd_stat_dict[frozenset({id1,id2})] = [half_len,full_len,half_count,full_count,num_segs_dnar,half_ibd_dnar,full_ibd_dnar,max_seg_cm_dnar]
        self.sex_dict = sex_dict
        self.age_dict = age_dict
        self.use_age_info = use_age_info
        self.min_rel_append_types = min_rel_append_types # Min number of other (less likely) relationship types to try when placing any given new person.
        self.max_rel_append_types = max_rel_append_types # Max number of other (less likely) relationship types to try when placing any given new person.
        self.twin_id_to_set = dict() # Map resid to index of twin set (twin_id_to_set[resid] = twin_set_index)
        self.twin_sets = dict() # Map index to set of twins (twin_sets[twin_set_index] = set({resid1,resid2,...}))
        self.del_twin_to_placed_twin = dict() # Map deleted twins to their twin who was placed in the ped (del_twin_to_placed_twin[resid] = placed_resid)
        self.placed_twin_to_del_twins = dict() # Map placed twin to the set of deleted twins (placed_twin_to_del_twin[resid] = [resid1,resid2,...] )
        self.traces = [] # Stores who was added to the pedigree in what order and all the ways that person was added
        self.plot_dir = plot_trace_path #'/Users/ejewett/Desktop/pb_plots/'
        self.plot_counter = 0

        self.unplaced_ids = list(self.pw_rels.keys())
        self.placed_ids = []

        self.incorrect_half_sib_dict = dict() # keys are a person of interest. Values are dicts mapping parent to the spurious half sibling
        self.incorrect_nth_avuncular_dict = dict() # keys are a person of interest. Values are dicts mapping grandparent to the spurious avuncular person
        self.incorrect_parent_child_dict = dict()
        self.incorrect_sibling_set = set()
        self.incorrect_placed_sibling_set = set()
        self.incorrect_unrelated_set = set()
        self.incorrect_related_set = set()

        if self.plot_dir:
            os.makedirs(self.plot_dir)


    # Find all twin/x-tuplets. Get two dictionaries: self.twin_id_to_set[resid] = twin_set_index , self.twin_sets[twin_set_index] = set({resid1,resid2,...})
    def find_twins(self):
        set_idx_ct = 0
        for id1,id2 in combinations( self.pw_rels.keys() , r=2 ):
            total_half_len = self.ibd_stat_dict[frozenset({id1,id2})]['total_half']
            total_full_len = self.ibd_stat_dict[frozenset({id1,id2})]['total_full']
            half_seg_count = self.ibd_stat_dict[frozenset({id1,id2})]['num_half']
            sex1 = self.sex_dict[id1]
            sex2 = self.sex_dict[id2]
            age1 = self.age_dict[id1]
            age2 = self.age_dict[id2]
            is_twin_pair = is_twin_pair(total_half_len, total_full_len, sex1, sex2)
            if is_twin_pair:
                set_idx1 = self.twin_id_to_set.get(id1)
                set_idx2 = self.twin_id_to_set.get(id2)
                if (set_idx1 is not None) and (set_idx2 is not None):
                    if set_idx1 == set_idx2:
                        continue
                    for resid in list(self.twin_sets[set_idx2]):
                        self.twin_id_to_set[resid] = set_idx1
                        self.twin_sets[set_idx1].add(resid)
                    del self.twin_sets[set_idx2]
                elif set_idx1 is not None:
                    self.twin_id_to_set[id2] = set_idx1
                    self.twin_sets[set_idx1].add(id2)
                elif set_idx2 is not None:
                    self.twin_id_to_set[id1] = set_idx2
                    self.twin_sets[set_idx2].add(id1)
                else: # Neither twin observed yet
                    self.twin_id_to_set[id1] = set_idx_ct
                    self.twin_id_to_set[id2] = set_idx_ct
                    self.twin_sets[set_idx_ct] = set({id1,id2})
                    set_idx_ct += 1

    # Delete all but one person from self.pw_rels and self.pw_log_likes. 
    # Save a dictionary mapping all x-tuplet-members in the pedigree to a
    # list of their excluded x-tuplets.
    def remove_twins(self, focal_id=None, keep_set=None):
        for set_idx , twin_set in iteritems(self.twin_sets):
            twins_in_keep_set = set()
            if keep_set:
                twins_in_keep_set = keep_set & twin_set
            if focal_id in twin_set:
                placed_id = focal_id
                twins_in_keep_set.add(focal_id)
                unplaced_ids = list(twin_set - twins_in_keep_set)
            else:
                if twins_in_keep_set:
                    twin_list = list(twins_in_keep_set)
                    placed_id = twin_list[0]
                    unplaced_ids = twin_set - twins_in_keep_set
                else:
                    twin_list = list(twin_set)
                    placed_id = twin_list[0]
                    unplaced_ids = twin_list[1:]
            if unplaced_ids:
                self.placed_twin_to_del_twins[placed_id] = unplaced_ids
                for unplaced_id in unplaced_ids:
                    self.del_twin_to_placed_twin[unplaced_id] = placed_id
                    all_ids = self.pw_rels.keys()
                    for resid in all_ids:
                        try:
                            del self.pw_rels[resid][unplaced_id]
                        except:
                            pass
                    try:
                        del self.pw_rels[unplaced_id]
                    except:
                        pass


    def enforce_twins(self):
        """
        set self.pw_log_likes and self.pw_rels to sibling for twin pairs
        """
        pairwise_log_likes = copy.deepcopy(self.pw_log_likes)
        for twin_set in self.twin_sets.values():
            for id1,id2 in combinations(twin_set, r=2):
                if (id1 not in self.pw_rels) or (id2 not in self.pw_rels):
                    continue
                self.pw_rels[id1][id2] = (1,1,2)
                self.pw_rels[id2][id1] = (1,1,2)
                enforce_rel_types(pairwise_log_likes=pairwise_log_likes[id1][id2] , deg_set=set({(1,1,2)}) , small_log_prob=SMALL_LOG_PROB)
                enforce_rel_types(pairwise_log_likes=pairwise_log_likes[id2][id1] , deg_set=set({(1,1,2)}) , small_log_prob=SMALL_LOG_PROB)
        self.pw_log_likes = pairwise_log_likes


    # Enforce sibling and parent-child relationships when they can be
    # corroborated by another sibling (or parent). Helps to speed up
    # pedigree building.
    def enforce_relationships( self ):

        pairwise_log_likes = deepcopy( self.pw_log_likes )

        # Find putative parent-child and sibling pairs
        parent_degs = set({(0,1,1),(1,0,1)})
        sibling_degs = set({(1,1,2)})
        putative_parents = defaultdict(set)
        putative_children = defaultdict(set)
        putative_siblings = defaultdict(set)
        for id1,rel_info in iteritems(self.pw_rels):
            id1_age = self.age_dict.get(id1)
            id1_sex = self.sex_dict.get(id1)
            if (not id1_age) or (not id1_sex):
                continue
            for id2,rel_deg in iteritems(rel_info):
                id2_age = self.age_dict.get(id2)
                id2_sex = self.sex_dict.get(id2)
                if (not id2_age) or (not id2_sex):
                    continue
                ibd_info = self.ibd_stat_dict[frozenset({id1,id2})]
                if rel_deg in parent_degs or (ibd_info['total_half'] > 0.95 * AUTO_GENOME_LENGTH and ibd_info['total_full'] < 0.25 * AUTO_GENOME_LENGTH):
                    if id1_age - id2_age >= 17: # Enforce parent-child to be at least 17 years apart (still allows for parent-child pairs < 17 years apart, but does not enforce them)
                        putative_parents[id2].add(id1)
                        putative_children[id1].add(id2)
                        enforce_rel_types( pairwise_log_likes=pairwise_log_likes[id2][id1] , deg_set=set({(1,0,1)}) , small_log_prob=SMALL_LOG_PROB )
                        enforce_rel_types( pairwise_log_likes=pairwise_log_likes[id1][id2] , deg_set=set({(0,1,1)}) , small_log_prob=SMALL_LOG_PROB )
                    elif id2_age - id1_age >= 17:
                        putative_parents[id1].add(id2)
                        putative_children[id2].add(id1)
                        enforce_rel_types( pairwise_log_likes=pairwise_log_likes[id1][id2] , deg_set=set({(1,0,1)}) , small_log_prob=SMALL_LOG_PROB )
                        enforce_rel_types( pairwise_log_likes=pairwise_log_likes[id2][id1] , deg_set=set({(0,1,1)}) , small_log_prob=SMALL_LOG_PROB )
                elif rel_deg in sibling_degs:
                    putative_siblings[id1].add(id2)
                    putative_siblings[id2].add(id1)

        # Enforce parent-child relationship if putatively parent-child
        # and there's at least one sibling with a putative parent-child
        # relationship to back it up
        for id1,parent_set in iteritems(putative_parents):
            id1_sibling_set = putative_siblings[id1]
            for id1_parent in list(parent_set):
                id1_parent_children = putative_children[id1_parent]
                true_sib_set = id1_sibling_set & id1_parent_children
                if len(true_sib_set) > 0:
                    enforce_rel_types( pairwise_log_likes=pairwise_log_likes[id1][id1_parent] , deg_set=set({(1,0,1)}) , small_log_prob=SMALL_LOG_PROB )
                    enforce_rel_types( pairwise_log_likes=pairwise_log_likes[id1_parent][id1] , deg_set=set({(0,1,1)}) , small_log_prob=SMALL_LOG_PROB )

            # Find unrelated people who both have an inferred parent-child relationship to another person
            # Enforce those parent-child relationships
            for p1,p2 in combinations( list(parent_set) , r=2 ):
                if self.pw_rels[p1][p2] != (float('inf'),float('inf'),None): # Make sure they're inferred to be unrelated
                    continue
                p1_children = putative_children[p1]
                p2_children = putative_children[p2]
                common_children = p1_children & p2_children
                for common_child in list(common_children):
                    enforce_rel_types( pairwise_log_likes=pairwise_log_likes[common_child][p1] , deg_set=set({(1,0,1)}) , small_log_prob=SMALL_LOG_PROB )
                    enforce_rel_types( pairwise_log_likes=pairwise_log_likes[p1][common_child] , deg_set=set({(0,1,1)}) , small_log_prob=SMALL_LOG_PROB )
                    enforce_rel_types( pairwise_log_likes=pairwise_log_likes[common_child][p2] , deg_set=set({(1,0,1)}) , small_log_prob=SMALL_LOG_PROB )
                    enforce_rel_types( pairwise_log_likes=pairwise_log_likes[p2][common_child] , deg_set=set({(0,1,1)}) , small_log_prob=SMALL_LOG_PROB )

        # Enforce siblings
        for id1,sib_set in iteritems(putative_siblings):
            for id2 in list(sib_set):
                full_ibd_len = self.ibd_stat_dict[frozenset({id1,id2})]['total_full']
                if full_ibd_len > 1000:
                    enforce_rel_types( pairwise_log_likes=pairwise_log_likes[id1][id2] , deg_set=set({(1,1,2)}) , small_log_prob=SMALL_LOG_PROB )
                    enforce_rel_types( pairwise_log_likes=pairwise_log_likes[id2][id1] , deg_set=set({(1,1,2)}) , small_log_prob=SMALL_LOG_PROB )

        self.pw_log_likes = pairwise_log_likes

    def add_relative(self, end_id, end_sex, end_age, rel_type, ped_list, placed_list, disallow_distant_half_rels, traces):
        """
        Add new_id to closest_rel_id, where rel_type specifies the relationship type from
        closest_rel_id to new_id. Return all pedigree objects that can be built with the
        specified relationship without overwriting an existing genotyped person in the pedigree.
        Do this by recursively branching out the specified relationship type from the
        specified "rel_id" in each pedigree object in ped_list.

        Args:
           rel_type: list of the form [up_meioses,down_meioses,num_common_ancs]
           ped_list: list of the form [[ped_obj,next_id_to_build_from,id_just_placed]].
                     When initializing, set iped_list = [[ped_obj,closest_rel_id,None]]
        """

        up, down, num_ancestors = rel_type

        new_ped_list = []
        new_traces = []
        for ind,ped_data in enumerate(ped_list):
            ped_obj, connect_id, prev_id = ped_data
            ped_copy = copy.deepcopy(ped_obj)

            ud = None
            action = None

            if up >= 1:
                ud = 'up'
                sex=None
                if up > 1:
                    action = 'continue'
                    terminate_sex = None
                elif up == 1 and down == 0: # Terminate at parent, so need to check that sex is consistent
                    action = 'terminate'
                    terminate_sex = end_sex
                elif up == 1 and num_ancestors == 2: # Only return one pedigree copy (with one parent to connect to).
                    action='bounce-2'
                    terminate_sex = None
                elif up == 1 and num_ancestors == 1: # Have to go through both parents. Return one pedigree copy for each parent.
                    action='bounce-1'
                    terminate_sex = None
                out_ped_objs = extend_up(
                    ped_obj=ped_copy,
                    connect_id=connect_id,
                    end_sex=terminate_sex,
                    action=action,
                    placed_list=placed_list
                )

            elif up == 0:
                ud = 'down'
                if down >= 1:
                    if down == 1:
                        terminate_sex = end_sex
                        if (prev_id is not None) and ( prev_id in ped_obj.down_pedigree_dict.get(connect_id,[]) ): # If we just bounced
                            if num_ancestors == 1:
                                action = 'bounce-terminate-1'
                            elif num_ancestors == 2:
                                action = 'bounce-terminate-2'
                        else:
                            action = 'terminate-1'
                    elif down > 1:
                        terminate_sex = None
                        if (prev_id is not None) and ( prev_id in ped_obj.down_pedigree_dict.get(connect_id,[]) ): # If we just bounced
                            if num_ancestors == 1:
                                action = 'bounce-continue-1'
                            elif num_ancestors == 2:
                                action = 'bounce-continue-2'
                        else:
                            action = 'continue-1'
                    out_ped_objs = extend_down(
                        ped_obj=ped_copy,
                        prev_id=prev_id,
                        connect_id=connect_id,
                        end_sex=terminate_sex,
                        action=action,
                        placed_list=placed_list,
                        disallow_distant_half_rels=disallow_distant_half_rels
                    )
                elif down == 0: # Then we terminate on this person
                    ud = 'terminate'
                    action='terminate'
                    if connect_id not in placed_list: # Don't overwrite an existing genotyped person
                        replaced_success = ped_copy.replace_individual(old_id=connect_id, new_id=end_id, new_sex=end_sex, new_age=end_age)
                        if replaced_success:
                            out_ped_objs = [[ped_copy,end_id,None]]
                        else:
                            out_ped_objs = []
                    else:
                        out_ped_objs = []

            prev_trace = traces[ind]
            for opo in out_ped_objs:
                tr_copy = copy.copy(prev_trace)
                tr_copy.append([connect_id,ud,action,prev_id])
                new_traces.append(tr_copy)
            new_ped_list.extend(out_ped_objs)

        if up == 0 and rel_type[1] == 0:
            return new_ped_list,new_traces
        else:
            rel_type = list(rel_type)
            if rel_type[0] > 0:
                rel_type[0] -= 1
            elif rel_type[1] > 0:
                rel_type[1] -= 1
            rel_type = tuple(rel_type)
            return self.add_relative(end_id=end_id, end_sex=end_sex, end_age=end_age, rel_type=rel_type, ped_list=new_ped_list, placed_list=placed_list , disallow_distant_half_rels=disallow_distant_half_rels, traces=new_traces)

    def add_relative_many_ways(self, rel_list, closest_rel_id, new_id, closest_rel_age, new_age, new_sex, disallow_distant_half_rels):
        """
        Add a new individual into a pedigree in many ways, relative to
        their closest relative in the pedigree.

        Args:
            rel_list:               list of relationship degrees (up,down,num_ancs) to add 
                                    new_id to closest_rel_id
            closest_rel_id:         id of the person in the pedigree to whom we are adding the new person
            closest_rel_age:        age of closest_rel_id
            new_id:                 ID of user being added to pedigree
            new_age:                age of new_id
            new_sex:                sex of new_id
        """

        new_ped_obj_list = []
        for rel_type in rel_list:
            if rel_type[2] is None:
                break

            age_consistent = False
            if closest_rel_age is not None and new_age is not None: # Check whether relationship type agrees with age ordering if pair are direct ancestor/descendants and age information exists.
                age_ambiguous = False
                if rel_type[1] == 0 and closest_rel_age <= new_age:
                    age_consistent = True
                elif rel_type[0] == 0 and closest_rel_age >= new_age:
                    age_consistent = True
                elif rel_type[0] > 0 and rel_type[1] > 0:
                    age_consistent = True
            else:
                age_ambiguous = True

            if age_ambiguous or age_consistent or (not self.use_age_info): # Don't add age-inconsistent people or pairs with greater than 2 degrees of separation (2 meioses).
                out_list, traces = self.add_relative(
                    end_id=new_id,
                    end_sex=new_sex,
                    end_age=new_age,
                    rel_type=rel_type,
                    ped_list=self.ped_obj_list,
                    placed_list=self.placed_ids,
                    disallow_distant_half_rels=disallow_distant_half_rels,
                    traces=[[]]*len(self.ped_obj_list)
                )
                self.traces.append([closest_rel_id, new_id, traces])
                new_ped_obj_list.extend(out_list)

        return new_ped_obj_list

    def make_draft_pedigree(self, 
        seed_id,
        max_radius=float('inf'),
        max_add_degree=2,
        ped_save_log_like_delta_fraction=0.333,
        ped_save_log_like_abs_fraction=0.1,
        #pw_like_factor=1e-10,
        seed_pedigree=None,
        validated_node_set=None,
        disallow_distant_half_rels=True,
        max_num_peds=50,
        recursion_depth=0,
        max_recursion_depth=5):

        """
        Iteratively make pedigrees, making all possible ones and keeping those
        with the highest likelihoods.

        Args:
           max_radius: Integer specifying a cutoff number of meioses. Any 
                       individual with a point-estimated relationship degree to
                       the seed_id exceeding max_radius is thrown out before
                       building the pedigree. Default is float('inf').
           max_rel_append_types: Set to a non-zero interger to override the
                                 max_rel_append_types for the pedigree builder.
           ped_save_log_like_fraction: We generate a list of pedigrees. We throw out
                                       unlikely ones. Throw out a pedigree only if
                                       it's likelihood is a factor ped_save_log_like_fraction
                                       smaller than the most likely pedigree.
           seed_pedigree:    A pedigree for part of the pedigree that has already been constructed.
        """

        self.ped_obj_list = [] # have to re-initialize
        self.traces = []

        if seed_pedigree is None:
            # If the seed_id is unspecified, find the person with the lowest
            # average degree to other people. This causes the pedigree to build
            # faster (and maybe a bit more accurately)
            self.unplaced_ids = list(self.pw_rels.keys())
            if len(self.unplaced_ids) == 0 and len(self.sex_dict) == 1: # if there's just one person
                seed_id = list(self.sex_dict.keys())[0]
            if seed_id is None:
                min_mean_deg = float('inf')
                seed_id = None
                for resid in self.unplaced_ids:
                    mean_deg = np.mean( [ val[0] + val[1] - none_zero(val[2]) + 1 for val in self.pw_rels[resid].values() if (float('inf') not in val) ] )
                    if mean_deg < min_mean_deg:
                        seed_id = resid
                        min_mean_deg = mean_deg
                if seed_id is None: # Can happen if all people are unrelated.
                    seed_id = self.unplaced_ids[0]
            else: # If seed_id is specified: r
                seed_id = self.del_twin_to_placed_twin.get(seed_id, seed_id) # if seed_id is a deleted twin, use the non-deleted twin id instead

            # Start the pedigree by adding one person and make lists of placed and unplaced IDs.
            ped_obj = PedigreeObject(
                up_pedigree_dict={},
                pairwise_log_likelihoods=self.pw_log_likes,
                ibd_stats=self.ibd_stat_dict,
                sex_dict=self.sex_dict,
                age_dict=self.age_dict,
                use_age_info=self.use_age_info
            )

            ped_obj.new_individual(new_id=seed_id, sex=self.sex_dict[seed_id], age=self.age_dict[seed_id])
            self.placed_ids = [seed_id] # Add ID to placed ID list.

            if len(self.unplaced_ids) == 0: # If there was only one person to place, return
                self.ped_obj_list.append([ped_obj,seed_id,None])
                self.ped_log_like_list = [ped_obj.pedigree_log_likelihood]
                return True

            self.unplaced_ids.remove(seed_id) # Remove ID from unplaced ID list.

        else: # if a seed pedigree is specified
            up_dict = dict()
            for uid,info in iteritems(seed_pedigree):
                sex,pid1,pid2 = info
                age = self.age_dict.get(uid,None)
                up_dict[uid] = [sex,age,pid1,pid2]
            ped_obj = trim_seed_pedigree(pedigree_up_dict=up_dict,
                       validated_node_set=validated_node_set,
                       pairwise_log_likelihoods=copy.deepcopy(self.pw_log_likes),
                       ibd_stats=copy.deepcopy(self.ibd_stat_dict),
                       sex_dict=copy.deepcopy(self.sex_dict),
                       age_dict=copy.deepcopy(self.age_dict),
                       use_age_info=self.use_age_info)
            self.unplaced_ids = list(set(self.pw_rels.keys()) - set(ped_obj.up_pedigree_dict.keys()))
            self.placed_ids = list(set(self.pw_rels.keys()) & set(ped_obj.up_pedigree_dict.keys()))
            min_abs_rel_deg = float('inf')
            for in_id in self.placed_ids:
                for out_id in self.unplaced_ids:
                    rel_deg = self.pw_rels[in_id][out_id]
                    abs_rel_deg = sum(rel_deg[:2]) - none_zero(rel_deg[2]) + 1
                    if abs_rel_deg < min_abs_rel_deg:
                        seed_id = in_id
                        min_abs_rel_deg = abs_rel_deg

        # Get degrees between placed and unplaced IDs (degree = number of meioses)
        # Only between people in and people outside of pedigree
        self.pw_degs = {}
        for in_id in self.placed_ids:
            for out_id in self.unplaced_ids:
                rel_deg = self.pw_rels[in_id][out_id]
                pair_log_likes = self.pw_log_likes[in_id].get(out_id)
                if pair_log_likes:
                    pair_log_like = pair_log_likes.get(rel_deg, -float('inf'))
                else:
                    pair_log_like = -float('inf')
                total_half = self.ibd_stat_dict[frozenset({in_id,out_id})]['total_half']
                self.pw_degs[(in_id, out_id)] = [in_id, out_id, pair_log_like, total_half, sum(self.pw_rels[in_id][out_id][:2]) - none_zero(self.pw_rels[in_id][out_id][2]) + 1]

        # Find individuals within max_radius meioses from the seed_id, and those outside of this radius
        keep_ids = set()
        remove_ids = set()
        for unplaced_id in self.unplaced_ids:
            if sum(self.pw_rels[seed_id][unplaced_id][:2]) <= max_radius:
                keep_ids.add(unplaced_id)
            else:
                remove_ids.add(unplaced_id)

        # Remove individuals outside of max_radius meioses from the target individual
        self.unplaced_ids = [subj_id for subj_id in self.unplaced_ids if subj_id in keep_ids]
        for in_id in self.placed_ids:
            for out_id in remove_ids:
                del self.pw_degs[(in_id, out_id)]

        self.ped_obj_list.append([ped_obj, seed_id, None])
        self.ped_log_like_list = [ped_obj.pedigree_log_likelihood]

        ret_val = True
        while self.unplaced_ids:

            # This prevents pedigrees from burgeoning when there is a chain of sparse relatives.
            if len(self.ped_obj_list) > max_num_peds:
                params = {
                    'seed_id': seed_id,
                    'max_radius': max_radius,
                    'max_add_degree': max_add_degree,
                    'use_age_info': self.use_age_info,
                    'min_rel_append_types': self.min_rel_append_types,
                    'max_rel_append_types': self.max_rel_append_types,
                }
                # raise Exception('ttam.bonsai prediction_failure %s' % str(params))
                break
            
            # Find the min degree
            min_deg = min([val[4] for val in self.pw_degs.values()])
            if min_deg > max_add_degree:
                break  # We don't add people beyond the add degree radius

            # Find the next pair to combine as the one that shares the most half IBD
            # break ties by choosing the one with the highest likelihood
            next_pair = (None,None)
            pair_max_half = 0
            pair_max_log_like = -float('inf')
            for pair_key,info in self.pw_degs.items():
                in_id,out_id,pair_log_like,total_half,deg = info
                if total_half > pair_max_half:
                    pair_max_half = total_half
                    pair_max_log_like = pair_log_like
                    next_pair = pair_key
                elif total_half == pair_max_half:
                    if pair_log_like > pair_max_log_like:
                        pair_max_log_like = pair_log_like
                        next_pair = pair_key

            closest_rel_id = next_pair[0]
            closest_rel_age = self.age_dict[closest_rel_id]
            new_id = next_pair[1]
            new_sex=self.sex_dict[new_id]
            new_age=self.age_dict.get(new_id)

            # Update ped_obj_list so that the most recently placed ID is the one we build
            # off of when we call self.add_relative()
            for ped_obj in self.ped_obj_list:
                ped_obj[1] = closest_rel_id
                ped_obj[2] = None

            # Add the new person to each existing pedigree in each of the most likely ways.
            # Compute the likelihood of the pedigree.
            # Throw out low-likelihood pedigrees and store the rest.
            rels = list(self.pw_log_likes[closest_rel_id][new_id].keys())
            rel_pair_log_likes = list(self.pw_log_likes[closest_rel_id][new_id].values())
            best_to_worst_order = list(reversed(np.argsort(rel_pair_log_likes)))

            # Get a list of relationship types to build
            rel_list = []
            seen_set = set()
            for ind in best_to_worst_order:
                log_like = rel_pair_log_likes[ind]
                rel_type = rels[ind]

                # Is the relationship type a distant half relative? If so, don't add it to the list.
                num_meioses = sum(rel_type[:2])
                num_ancs = rel_type[2]
                is_distant_half_rel = False
                if num_meioses > 2 and num_ancs == 1:
                    is_distant_half_rel = True
                if disallow_distant_half_rels and is_distant_half_rel:
                    continue

                ordered_rel_type = ( min(rel_type[:2]) , max(rel_type[:2]) , rel_type[2] )
                if log_like > SMALL_LOG_PROB and ((len(seen_set) < self.max_rel_append_types) or (len(seen_set) == self.max_rel_append_types and ordered_rel_type in seen_set)):
                    rel_list.append(rel_type)
                    seen_set.add(ordered_rel_type)
                else:
                    break

            new_ped_obj_list = self.add_relative_many_ways(
                rel_list=rel_list,
                closest_rel_id=closest_rel_id,
                new_id=new_id,
                closest_rel_age=closest_rel_age,
                new_age=new_age,
                new_sex=new_sex,
                disallow_distant_half_rels=disallow_distant_half_rels
            )            
                
            if len(new_ped_obj_list) == 0:
                ret_val = False
                break

            for in_ind in self.placed_ids:
                del self.pw_degs[(in_ind,new_id)]
            self.unplaced_ids.remove(new_id)

            for unplaced_id in self.unplaced_ids:
                rel_deg = self.pw_rels[new_id][unplaced_id]
                pair_log_likes = self.pw_log_likes[new_id].get(unplaced_id)
                if pair_log_likes:
                    pair_log_like = pair_log_likes.get(rel_deg,-float('inf'))
                else:
                    pair_log_like = -float('inf')
                total_half = self.ibd_stat_dict[frozenset({new_id,unplaced_id})]['total_half']
                self.pw_degs[(new_id, unplaced_id)] = [new_id, unplaced_id, pair_log_like, total_half, sum(self.pw_rels[new_id][unplaced_id][:2]) - none_zero(self.pw_rels[new_id][unplaced_id][2]) + 1]
            self.placed_ids.append(new_id)

            # Find the highest likelihood
            temp_ped_log_like_list = [ped_data[0].pedigree_log_likelihood for ped_data in new_ped_obj_list]
            new_log_likes = sorted(np.unique(temp_ped_log_like_list), reverse=True)

            # Dynamically set the log likelihood cutoff below which to throw out pedigrees
            log_like_cutoff = new_log_likes[0]
            # Minimum log ratio of next log like and prev log like. Find where the
            # likelihood drops by a factor of ped_save_log_like_fraction (default
            # is a factor of 3ish) or more and throw out subsequent likelihoods.
            ped_log_delta = np.log(ped_save_log_like_delta_fraction)
            ped_log_abs = np.log(ped_save_log_like_abs_fraction)
            max_ped_log_like = new_log_likes[0]
            for next_ind in range(1, len(new_log_likes)):
                next_log_like = new_log_likes[next_ind]
                if (next_log_like >= log_like_cutoff + ped_log_delta)  and (next_log_like >= max_ped_log_like + ped_log_abs): # Only keep likelihoods that are at least a fraction of the previous one, and a fraction of the most likely one
                    log_like_cutoff = next_log_like
                else:
                    break

            self.ped_obj_list = []
            self.ped_log_like_list = []
            for ped_data in new_ped_obj_list:
                ped_log_like = ped_data[0].pedigree_log_likelihood
                if ped_log_like >= log_like_cutoff:
                    self.ped_obj_list.append(ped_data)
                    self.ped_log_like_list.append(ped_log_like)

            logging.debug("Num pedigrees: %d" % len(self.ped_obj_list))
            logging.debug(str(self.placed_ids))

            # Delete duplicate pedigree objects
            del_inds = []
            seen_pedigress = set()
            for idx, (po, _, _) in enumerate(self.ped_obj_list):
                ped = frozenset((resid, frozenset(info[2:])) for resid, info in iteritems(po.up_pedigree_dict))
                if ped in seen_pedigress:
                    del_inds.append(idx)
                else:
                    seen_pedigress.add(ped)
            # Delete from the end so index doesn't change as we delete
            for idx in reversed(del_inds):
                del self.ped_obj_list[idx]
                del self.ped_log_like_list[idx]

        # check whether anyone got placed in a weird position and, if so, re-run
        most_likely_ped_obj = self.ped_obj_list[np.argmax(self.ped_log_like_list)][0]
        self.incorrect_half_sib_dict = dict() # keys are a person of interest. Values are dicts mapping parent to the spurious half sibling
        self.incorrect_nth_avuncular_dict = dict() # keys are a person of interest. Values are dicts mapping grandparent to the spurious avuncular person
        self.incorrect_parent_child_dict = dict()
        self.incorrect_sibling_set = set()
        self.incorrect_placed_sibling_set = set()
        self.incorrect_unrelated_set = set() # people who share no IBD placed as close relatives
        self.incorrect_related_set = set() # people who share substantial IBD placed as unrelated
        for rel1,rel_info in iteritems(most_likely_ped_obj.rels):
            for rel2,rel_deg in iteritems(rel_info):
                if (not most_likely_ped_obj.is_genotyped(rel1)) or (not most_likely_ped_obj.is_genotyped(rel2)):
                    continue
                if rel1 == rel2:
                    continue
                if rel_deg == (0,1,1) or rel_deg == (1,0,1):
                    total_half = self.ibd_stat_dict[frozenset({rel1,rel2})]['total_half']
                    if total_half < 0.8 * AUTO_GENOME_LENGTH:
                        if rel_deg == (1,0,1):
                            child_id = rel1
                            parent_id = rel2
                        else:
                            child_id = rel2
                            parent_id = rel1
                        if parent_id not in self.incorrect_parent_child_dict:
                            self.incorrect_parent_child_dict[parent_id] = set()
                        self.incorrect_parent_child_dict[parent_id].add(child_id)
                        possible_half_sib_list = [uid for uid,deg in iteritems(most_likely_ped_obj.rels[child_id]) if deg == (1,1,1) and most_likely_ped_obj.is_genotyped(uid)]
                        spurious_half_sib_list = [uid for uid in possible_half_sib_list if parent_id in most_likely_ped_obj.up_pedigree_dict[uid][2:]] # if they have parent_id as a parent, who we suspect is not a parent
                        for spurious_half_sib_id in spurious_half_sib_list:
                            if child_id not in self.incorrect_half_sib_dict:
                                self.incorrect_half_sib_dict[child_id] = dict()
                            if parent_id not in self.incorrect_half_sib_dict[child_id]:
                                self.incorrect_half_sib_dict[child_id][parent_id] = set()
                            self.incorrect_half_sib_dict[child_id][parent_id].add(spurious_half_sib_id)
                elif rel_deg == (1,1,2):
                    total_full = self.ibd_stat_dict[frozenset({rel1,rel2})]['total_full']
                    if total_full < 100:
                        self.incorrect_placed_sibling_set.add(frozenset({rel1,rel2}))
                elif rel_deg[2] is not None:
                    if self.pw_rels[rel1][rel2][2] is None:
                        if rel_deg[0] + rel_deg[1] - rel_deg[2] + 1 <= 5: # A degree of 5 is consistent with a 1/2% chance of sharing no IBD given that you are truly related
                            self.incorrect_unrelated_set.add(frozenset({rel1,rel2}))
                elif rel_deg[2] == 1 and (rel1 != rel2) and (rel_deg[0] == 0 or rel_deg[1] == 0):
                    total_half = self.ibd_stat_dict[frozenset({rel1,rel2})]['total_half']
                    total_full = self.ibd_stat_dict[frozenset({rel1,rel2})]['total_full']
                    if total_half == 0 and total_full == 0:
                        if rel_deg[1] == 0:
                            descendant_id = rel1
                            ancestor_id = rel2
                        elif rel_deg[0] == 0:
                            descendant_id = rel2
                            ancestor_id = rel1
                        ancestor_child_ids = most_likely_ped_obj.down_pedigree_dict[ancestor_id][2:]
                        spurious_nth_avuncular_list = [uid for uid in ancestor_child_ids if most_likely_ped_obj.is_genotyped(uid) and most_likely_ped_obj.rels[descendant_id][uid][2] == 2] # we don't need the case num_ancs == 1 because then the relative can have another branch that is not related to descendant_id
                        for spurious_nth_avuncular_id in spurious_nth_avuncular_list:
                            if descendant_id not in self.incorrect_nth_avuncular_dict:
                                self.incorrect_nth_avuncular_dict[descendant_id] = dict()
                            if ancestor_id not in self.incorrect_nth_avuncular_dict[descendant_id]:
                                self.incorrect_nth_avuncular_dict[descendant_id][ancestor_id] = set()
                            self.incorrect_nth_avuncular_dict[descendant_id][ancestor_id].add(spurious_nth_avuncular_id)
                elif (self.pw_rels[rel1][rel2] == (1,1,2)) and (rel_deg != (1,1,2)): # Test for sibs placed as something else
                    total_half = self.ibd_stat_dict[frozenset({rel1,rel2})]['total_half']
                    total_full = self.ibd_stat_dict[frozenset({rel1,rel2})]['total_full']
                    if total_half >= 1910 and total_full >= 135:
                        self.incorrect_sibling_set.add(frozenset({rel1,rel2}))

        inconsistent_sexes, inconsistent_id = most_likely_ped_obj.inconsistent_sexes()

        if (inconsistent_sexes or self.incorrect_parent_child_dict or self.incorrect_placed_sibling_set or self.incorrect_half_sib_dict or self.incorrect_nth_avuncular_dict or self.incorrect_unrelated_set or self.incorrect_related_set or self.incorrect_sibling_set) and recursion_depth < max_recursion_depth: # If some relationship types were odd, rerun recursively
            pairwise_relationships = self.pw_rels
            pairwise_log_likelihoods = self.pw_log_likes

            # disallow spurious related pairs
            for pair_set in self.incorrect_unrelated_set:
                rel1,rel2 = list(pair_set)
                for deg in pairwise_log_likelihoods[rel1][rel2].keys():
                    if deg[2] is not None:
                        pairwise_log_likelihoods[rel1][rel2][deg] = SMALL_LOG_PROB
                        pairwise_log_likelihoods[rel2][rel1][deg] = SMALL_LOG_PROB
                update_pw_rel(rel1,rel2,pairwise_log_likelihoods,pairwise_relationships)

            # disallow spurious unrelated pairs
            for pair_set in self.incorrect_related_set:
                rel1,rel2 = list(pair_set)
                unrel_deg = (float('inf'), float('inf'), None)
                pairwise_log_likelihoods[rel1][rel2][unrel_deg] = SMALL_LOG_PROB
                pairwise_log_likelihoods[rel2][rel1][unrel_deg] = SMALL_LOG_PROB
                update_pw_rel(rel1,rel2,pairwise_log_likelihoods,pairwise_relationships)
                                
            # disallow spurious parent-child pairs
            for parent_id,kid_id_set in self.incorrect_parent_child_dict.items():
                for kid_id in kid_id_set:
                    pairwise_log_likelihoods[kid_id][parent_id][(1,0,1)] = SMALL_LOG_PROB
                    pairwise_log_likelihoods[parent_id][kid_id][(0,1,1)] = SMALL_LOG_PROB
                    update_pw_rel(kid_id,parent_id,pairwise_log_likelihoods,pairwise_relationships)

            # disallow spurious full sibling pairs
            for sib_id_set in self.incorrect_placed_sibling_set:
                kid1_id, kid2_id = list(sib_id_set)
                pairwise_log_likelihoods[kid1_id][kid2_id][(1,1,2)] = SMALL_LOG_PROB
                pairwise_log_likelihoods[kid2_id][kid1_id][(1,1,2)] = SMALL_LOG_PROB
                update_pw_rel(kid1_id,kid2_id,pairwise_log_likelihoods,pairwise_relationships)

            # disallow full sibling pairs placed as something else
            for pair_set in self.incorrect_sibling_set:
                rel1,rel2 = list(pair_set)
                for deg in pairwise_log_likelihoods[rel1][rel2].keys():
                    if deg != (1,1,2):
                        pairwise_log_likelihoods[rel1][rel2][deg] = SMALL_LOG_PROB
                        pairwise_log_likelihoods[rel2][rel1][deg] = SMALL_LOG_PROB
                update_pw_rel(rel1,rel2,pairwise_log_likelihoods,pairwise_relationships)

            # disallow parent and half sib relationships for spurious half-sib-parent sets.
            for uid,info in iteritems(self.incorrect_half_sib_dict):
                for parent_id,spurious_half_sib_set in iteritems(info):
                    for spurious_half_sib_id in spurious_half_sib_set:
                        pairwise_log_likelihoods[uid][spurious_half_sib_id][(1,1,1)] = SMALL_LOG_PROB
                        pairwise_log_likelihoods[spurious_half_sib_id][uid][(1,1,1)] = SMALL_LOG_PROB
                        update_pw_rel(uid,spurious_half_sib_id,pairwise_log_likelihoods,pairwise_relationships)

                        pairwise_log_likelihoods[uid][parent_id][(1,0,1)] = SMALL_LOG_PROB
                        pairwise_log_likelihoods[parent_id][uid][(0,1,1)] = SMALL_LOG_PROB
                        update_pw_rel(uid,parent_id,pairwise_log_likelihoods,pairwise_relationships)

                        if spurious_half_sib_id in self.incorrect_parent_child_dict[parent_id]:
                            pairwise_log_likelihoods[spurious_half_sib_id][parent_id][(1,0,1)] = SMALL_LOG_PROB
                            pairwise_log_likelihoods[parent_id][spurious_half_sib_id][(0,1,1)] = SMALL_LOG_PROB
                            update_pw_rel(spurious_half_sib_id,parent_id,pairwise_log_likelihoods,pairwise_relationships)

            # disallow direct ancestors and spurious nth avuncular individuals.
            for uid,info in iteritems(self.incorrect_nth_avuncular_dict):
                for ancestor_id,spurious_nth_avuncular_set in iteritems(info):
                    for spurious_nth_avuncular_id in spurious_nth_avuncular_set:
                        deg = most_likely_ped_obj.rels[uid][spurious_nth_avuncular_id]
                        rev_deg = (deg[1],deg[0],deg[2])
                        pairwise_log_likelihoods[uid][spurious_nth_avuncular_id][deg] = SMALL_LOG_PROB
                        pairwise_log_likelihoods[spurious_nth_avuncular_id][uid][rev_deg] = SMALL_LOG_PROB
                        update_pw_rel(uid,spurious_nth_avuncular_id,pairwise_log_likelihoods,pairwise_relationships)

                        deg = most_likely_ped_obj.rels[uid][ancestor_id]
                        rev_deg = (deg[1],deg[0],deg[2])
                        pairwise_log_likelihoods[uid][ancestor_id][deg] = SMALL_LOG_PROB
                        pairwise_log_likelihoods[ancestor_id][uid][rev_deg] = SMALL_LOG_PROB
                        update_pw_rel(uid,ancestor_id,pairwise_log_likelihoods,pairwise_relationships)

                        if spurious_nth_avuncular_id in self.incorrect_parent_child_dict.get(ancestor_id,[]):
                            pairwise_log_likelihoods[spurious_nth_avuncular_id][ancestor_id][(1,0,1)] = SMALL_LOG_PROB
                            pairwise_log_likelihoods[ancestor_id][spurious_nth_avuncular_id][(0,1,1)] = SMALL_LOG_PROB
                            update_pw_rel(spurious_nth_avuncular_id,ancestor_id,pairwise_log_likelihoods,pairwise_relationships)

            new_pb = PedigreeBuilder(pairwise_relationships = pairwise_relationships,
                                     pairwise_log_likelihoods = pairwise_log_likelihoods,
                                     ibd_stat_dict = self.ibd_stat_dict,
                                     sex_dict = self.sex_dict,
                                     age_dict = self.age_dict,
                                     use_age_info = self.use_age_info,
                                     min_rel_append_types = self.min_rel_append_types,
                                     max_rel_append_types = self.max_rel_append_types)

            new_pb.enforce_relationships()

            new_pb.make_draft_pedigree(seed_id=seed_id, 
                                       max_radius=max_radius, 
                                       max_add_degree=max_add_degree, 
                                       ped_save_log_like_delta_fraction=0.0001,
                                       ped_save_log_like_abs_fraction=0.00001,
                                       seed_pedigree=seed_pedigree,
                                       validated_node_set=validated_node_set,
                                       disallow_distant_half_rels=disallow_distant_half_rels, # don't disallow distant half rels. This can eff things up by overly restricting the search space
                                       max_num_peds=max_num_peds,
                                       recursion_depth=recursion_depth+1,
                                       max_recursion_depth=max_recursion_depth)

            self.ped_obj_list = new_pb.ped_obj_list
            self.ped_log_like_list = new_pb.ped_log_like_list
            self.unplaced_ids = new_pb.unplaced_ids
            self.placed_ids = new_pb.placed_ids
  
        # we might not have placed every individual in the twin dicts. Restrict to placed individuals
        example_ped_obj = self.ped_obj_list[0][0]
        all_id_list = example_ped_obj.all_ids
        all_id_set = set(all_id_list)
        self.twin_sets = {twin_index : twin_set for twin_index,twin_set in iteritems(self.twin_sets) if len(twin_set & all_id_set) > 0} # Map index to set of twins (twin_sets[twin_set_index] = set({resid1,resid2,...}))
        self.del_twin_to_placed_twin = {del_twin : placed_twin for del_twin,placed_twin in iteritems(self.del_twin_to_placed_twin) if placed_twin in all_id_set} # Map deleted twins to their twin who was placed in the ped (del_twin_to_placed_twin[resid] = placed_resid)
        self.placed_twin_to_del_twins = {placed_twin : del_twins for placed_twin,del_twins in iteritems(self.placed_twin_to_del_twins) if placed_twin in all_id_set} # Map placed twin to the set of deleted twins (placed_twin_to_del_twin[resid] = [resid1,resid2,...] )

        return ret_val


def trim_seed_pedigree(pedigree_up_dict,
                       validated_node_set,
                       pairwise_log_likelihoods,
                       ibd_stats,
                       sex_dict,
                       age_dict,
                       use_age_info):
    """
    Take a validated pedigree reduce it to just the parts connecting
    genotyped people and any spouses along the paths connecting
    genotyped people.
    """

    ped_obj = PedigreeObject(up_pedigree_dict=pedigree_up_dict,
                             pairwise_log_likelihoods=pairwise_log_likelihoods,
                             ibd_stats=ibd_stats,
                             sex_dict=sex_dict,
                             age_dict=age_dict,
                             use_age_info=use_age_info)
    ped_obj_copy = copy.deepcopy(ped_obj)

    # keep all:
    #    genotyped nodes
    #    people on a path between two genotyped nodes
    #    descendants of any of the above people
    #    both parents of any of the above individuals if they have at least one parent who is one of the above individuals
    # remove 
    #    dangly ancestral branches that don't attach two genotyped nodes.
    #    These get in the way of the big pedigree builder and we can't usually extend branches up the correct lineage in the small builder either.
    genotyped_and_validated_set = {uid for uid in pedigree_up_dict.keys() if (not (isinstance(uid,int) and uid < 0)) and uid in validated_node_set}
    ped_obj_copy.keep_nodes(genotyped_and_validated_set)
    backbone_node_id_set = set(ped_obj_copy.up_pedigree_dict.keys()) | set(ped_obj_copy.down_pedigree_dict.keys()) # all genotyped people and those connecting genotyped people
    keep_set = copy.deepcopy(backbone_node_id_set)
    for uid in backbone_node_id_set:
        sex,age,pid1,pid2 = pedigree_up_dict[uid]
        if len({pid1,pid2} & backbone_node_id_set):
            keep_set |= {pid1,pid2} # if at least one parent of uid is in the genotyped_and_leaf_set, make sure both parents are in the set
    ped_obj.keep_nodes(keep_set)
    return ped_obj


def update_pw_rel(uid1,uid2,pw_log_likes,pw_rels):
    max_log_like = -float('inf')
    new_deg = None
    for deg,log_like in iteritems(pw_log_likes[uid1][uid2]):
        if log_like > max_log_like:
            max_log_like = log_like
            new_deg = deg
    rev_new_deg = (new_deg[1],new_deg[0],new_deg[2])
    pw_rels[uid1][uid2] = new_deg
    pw_rels[uid2][uid1] = rev_new_deg


def check_parent_symmetry(subj_id, ped_obj):
    try:
        info = ped_obj.up_pedigree_dict[subj_id]
        id1,id2 = info[2:]
    except:
        raise ValueError("Subject has fewer than two parents.")
    id1_parents = ped_obj.up_pedigree_dict.get(id1,[])[2:]
    id2_parents = ped_obj.up_pedigree_dict.get(id2,[])[2:]
    id1_kids = ped_obj.down_pedigree_dict.get(id1,[])[2:]
    id2_kids = ped_obj.down_pedigree_dict.get(id2,[])[2:]
    id1_sex = ped_obj.down_pedigree_dict[id1][0]
    id2_sex = ped_obj.down_pedigree_dict[id2][0]
    if set(id1_parents) == set(id2_parents) and set(id1_kids) == set(id2_kids) and (not id1_sex) and (not id2_sex):
        symmetric = True
    else:
        symmetric = False
    return symmetric

def extend_up(ped_obj, connect_id, end_sex, action, placed_list):
    """
    connect_id: last person added to pedigree

    Extend up one generation and return a list of pedigrees, one for each extension.
        If no parents exist, create one to extend up from (create the other one, but don't extend to it in order to reduce the state space... should be okay now that sex doesn't matter here).
        If one parent already exists, extend up to that parent and also add another one to extend from.
        If both parents exist, extend up to both of them.
        If terminating at a parent, esure sex is correct.
    """
    info = ped_obj.up_pedigree_dict[connect_id]
    parents = info[2:]
    out_list = []
    if len(parents) == 0:
        parent1_id = ped_obj.add_parent_for_child(child_id=connect_id, parent_sex=None)
        parent2_id = ped_obj.add_parent_for_child(child_id=connect_id, parent_sex=None)
        out_list.append([ped_obj, parent1_id, connect_id])
    elif len(parents) == 1:
        parent1_id = parents[0]
        parent2_id = ped_obj.add_parent_for_child( child_id=connect_id , parent_sex=None )
        parent1_sex = ped_obj.down_pedigree_dict[parent1_id][0]
        parent2_sex = ped_obj.down_pedigree_dict[parent2_id][0]
        if action == 'terminate':
            if parent1_sex == end_sex or ( parent2_sex == ped_obj.opposite_sex[end_sex] ):
                parent_id = parent1_id
            elif parent2_sex == end_sex or ( parent1_sex == ped_obj.opposite_sex[end_sex] ):
                parent_id = parent2_id
            if parent_id not in placed_list:
                out_list.append( [ ped_obj , parent_id , connect_id ] )
        elif action == 'continue' or action=='bounce-1':
            symmetric_parents = check_parent_symmetry( connect_id , ped_obj )
            if symmetric_parents: # If parents are symmetric.
                out_list.append( [ ped_obj , parent1_id , connect_id ] )
            else:
                if not (action=='continue' and (parent1_id in placed_list)): # Don't continue through a genotyped person
                    out_list.append( [ ped_obj , parent1_id , connect_id ] )
                if not (action=='continue' and (parent2_id in placed_list)): # Don't continue through a genotyped person
                    ped_copy = copy.deepcopy(ped_obj)
                    out_list.append( [ ped_copy , parent2_id , connect_id ] )
        elif action=='bounce-2':
            out_list.append( [ ped_obj , parent1_id , connect_id ] ) # TODO: I believe we handle connecting the next ID to both parents of connect_id in the extend_down function, but if we're getting spurious half reltives, this might be a candidate for where that's happening.
    elif len(parents) == 2:
        symmetric_parents = check_parent_symmetry( connect_id , ped_obj )
        parent1_id,parent2_id = parents
        parent1_sex = ped_obj.down_pedigree_dict[parent1_id][0]
        parent2_sex = ped_obj.down_pedigree_dict[parent2_id][0]
        if action == 'terminate':
            if parent1_sex is None and parent2_sex is None:
                if symmetric_parents: # Then it doesn't matter who we choose
                    out_list.append( [ ped_obj , parent1_id , connect_id ] )
                else: # Then we need one copy for each parent
                    if parent1_id not in placed_list: # Placed IDs are genotyped people. Don't extend up through a genotyped person
                        out_list.append( [ ped_obj , parent1_id , connect_id ] )
                    if parent2_id not in placed_list: # Placed IDs are genotyped people. Don't extend up through a genotyped person
                        ped_copy = copy.deepcopy(ped_obj)
                        out_list.append( [ ped_copy , parent2_id , connect_id ] )
            elif parent1_sex == end_sex or ( parent2_sex == ped_obj.opposite_sex[end_sex] ):
                if parent1_id not in placed_list:
                    out_list.append( [ ped_obj , parent1_id , connect_id ] )
            elif parent2_sex == end_sex or ( parent1_sex == ped_obj.opposite_sex[end_sex] ):
                if parent2_id not in placed_list:
                    out_list.append( [ ped_obj , parent2_id , connect_id ] )
        elif action == 'continue' or action == 'bounce-1':
            if symmetric_parents:
                out_list.append( [ ped_obj , parent1_id , connect_id ] )
            else:
                out_list.append( [ ped_obj , parent1_id , connect_id ] )
                ped_copy = copy.deepcopy(ped_obj)
                out_list.append( [ ped_copy , parent2_id , connect_id ] )
        elif action == 'bounce-2':
            out_list.append( [ ped_obj , parent1_id , connect_id ] )
    return out_list


def extend_down(ped_obj, prev_id, connect_id, end_sex, action, placed_list, disallow_distant_half_rels):
    out_list = []
    if connect_id not in ped_obj.down_pedigree_dict:
        child_id = ped_obj.add_child_for_parent( parent_id=connect_id , child_sex=end_sex )
        child_ids = [child_id]
        ped_obj.add_parent_for_child( child_id=child_id , parent_sex=None )
        out_list.append( [ ped_obj , child_id , connect_id ] )
    else:
        info = ped_obj.down_pedigree_dict[connect_id]
        sex = info[0]

        child_ids = info[2:]
        child_info_dict = { child_id : ped_obj.up_pedigree_dict[child_id] for child_id in child_ids }
        child_id_parent_sets = { child_id : frozenset( child_info_dict[child_id][2:] ) for child_id in child_ids }
        parent_to_child_dict = defaultdict(list)
        for child_id,parent_set in iteritems(child_id_parent_sets):
            parent_to_child_dict[parent_set].append(child_id)

        prev_id_parent_set = frozenset( ped_obj.up_pedigree_dict.get(prev_id,[])[2:] )
        any_genotyped_child = np.any([ped_obj.is_genotyped(child_id) for child_id in child_ids])

        if action=='bounce-terminate-2' or action=='bounce-continue-2':
            possible_parent_sets = [prev_id_parent_set]
        elif action=='bounce-terminate-1' or action=='bounce-continue-1':
            ped_copy = copy.deepcopy(ped_obj) # Make a child from the parent and a new parent we haven't seen before.
            new_child_id = ped_copy.add_child_for_parent( parent_id=connect_id , child_sex=end_sex )
            new_parent_id = ped_copy.add_parent_for_child( child_id=new_child_id , parent_sex=None )
            out_list.append( [ ped_copy , new_child_id , connect_id ] )
            new_parent_set = frozenset({connect_id , new_parent_id})
            possible_parent_sets = set(parent_to_child_dict.keys()) - set([prev_id_parent_set])
            possible_parent_sets -= new_parent_set
        elif action=='terminate-1':
            possible_parent_sets = set(parent_to_child_dict.keys())
            if (len(possible_parent_sets) == 0) or (not disallow_distant_half_rels) or (disallow_distant_half_rels and ped_obj.is_genotyped(connect_id)) or (disallow_distant_half_rels and any_genotyped_child):
                ped_copy = copy.deepcopy(ped_obj) # Make a child from the parent and a new parent we haven't seen before.
                new_child_id = ped_copy.add_child_for_parent( parent_id=connect_id , child_sex=end_sex )
                new_parent_id = ped_copy.add_parent_for_child( child_id=new_child_id , parent_sex=None )
                out_list.append( [ ped_copy , new_child_id , connect_id ] )
        elif action=='continue-1':
            possible_parent_sets = set(parent_to_child_dict.keys())
            if (len(possible_parent_sets) == 0) or (not disallow_distant_half_rels) or (disallow_distant_half_rels and ped_obj.is_genotyped(connect_id)) or (disallow_distant_half_rels and any_genotyped_child):
                ped_copy = copy.deepcopy(ped_obj) # Make a child from the parent and a new parent we haven't seen before.
                new_child_id = ped_copy.add_child_for_parent( parent_id=connect_id , child_sex=end_sex )
                new_parent_id = ped_copy.add_parent_for_child( child_id=new_child_id , parent_sex=None )
                out_list.append( [ ped_copy , new_child_id , connect_id ] )


        for parent_set in possible_parent_sets:
            other_parent_id = list(parent_set - set([connect_id]))[0]
            ped_copy = copy.deepcopy(ped_obj) # Make a child we haven't seen before.
            new_child_id = ped_copy.add_child_for_parent( parent_id=connect_id , child_sex=end_sex )
            ped_copy.connect_parent_child( child_id=new_child_id , parent_id=other_parent_id )
            out_list.append( [ ped_copy , new_child_id , connect_id ] )
            for child_id in parent_to_child_dict[parent_set]:
                if child_id == prev_id: # FIXED 5/3/2019
                    continue # FIXED 5/3/2019
                if child_id in placed_list:
                    continue
                if ( action=='terminate-1' or action=='bounce-terminate-1' or action=='bounce-terminate-2' ) and (child_id not in placed_list): #If they're ungenotyped
                    child_sex = child_info_dict[child_id][0]
                    if child_sex == end_sex:
                        ped_copy = copy.deepcopy(ped_obj)
                        out_list.append( [ ped_copy , child_id , connect_id ] )
                elif action=='continue-1' or action=='bounce-continue-1' or action=='bounce-continue-2':
                    ped_copy = copy.deepcopy(ped_obj)
                    out_list.append( [ ped_copy , child_id , connect_id ] )

    return out_list



def check_pred_deg_likes(
    focal_id,
    pw_rels,
    pw_log_likes,
    ped_pw_rels,
    placed_gt_id_list,
    age_dict,
    radius_deg=5,
    deg_diff=2,
    user_error_detected=False,
    throw_exception=False
):
    """
    Check whether all pairs closer than 1st cousins within a radius of first cousin 
    around focal_id have predicted bonsai degrees that are close to their ML degrees.
    """

    unrel_deg = (float('inf'),float('inf'),None)
    parent_deg_set = {(1,0,1),(0,1,1)}

    bonsai_likelihood_agreement = True
    warn_only = True
    unlikely_pair_info_list = []
    for id1,id2 in combinations(placed_gt_id_list,r=2):

        # check pw degree of id1 and id2 to focal. If too far away, ignore so we 
        # don't fail on trees if a distant relationship is wrong.
        if id1 == focal_id:
            deg1 = (0,0,2)
        else:
            deg1 = pw_rels[focal_id][id1]

        if id2 == focal_id:
            deg2 = (0,0,2)
        else:
            deg2 = pw_rels[focal_id][id2]

        if deg1 == unrel_deg:
            abs_deg1 = float('inf')
        else:
            abs_deg1 = deg1[0] + deg1[1] - deg1[2] + 1

        if deg2 == unrel_deg:
            abs_deg2 = float('inf')
        else:
            abs_deg2 = deg2[0] + deg2[1] - deg2[2] + 1

        if (focal_id not in {id1,id2}) and ((abs_deg1 > radius_deg) or (abs_deg2 > radius_deg)):
            continue

        # check degree between id1 and id2. Ignore if id1 and id2 are too far apart so we 
        # don't fail on trees if a distant rel is wrong.
        pw_deg = pw_rels[id1][id2]
        pred_deg = ped_pw_rels[id1].get(id2,unrel_deg)
        
        if pw_deg == unrel_deg:
            abs_pw_deg = float('inf')
        else:
            abs_pw_deg = pw_deg[0] + pw_deg[1] - pw_deg[2] + 1

        if pred_deg == unrel_deg:
            abs_pred_deg = float('inf')
        else:
            abs_pred_deg = pred_deg[0] + pred_deg[1] - pred_deg[2] + 1

         # allow inf so we can test if unrelated are being placed together
        if (focal_id not in {id1,id2}) and (abs_pw_deg > radius_deg) and (abs_pw_deg != float('inf')):
            continue

        # check whether the original pw estimate may have been wrong
        ambiguous_pw_deg = False
        if pw_deg in parent_deg_set:
            age1 = age_dict.get(id1)
            age2 = age_dict.get(id2)
            if (not isinstance(age1,int)) or (not isinstance(age2,int)):
                ambiguous_pw_deg = True
            elif abs(age1 - age2) <= MIN_PARENT_CHILD_AGE_DIFF:
                ambiguous_pw_deg = True
        # can add more cases ...

        if ambiguous_pw_deg and {pw_deg,pred_deg} == parent_deg_set:
            continue
        elif ({pw_deg,pred_deg} & parent_deg_set) and pw_deg != pred_deg:
            bonsai_likelihood_agreement = False
            warn_only = False
            unlikely_pair_info_list.append([focal_id,id1,id2,pw_deg,pred_deg])
        elif (1,1,2) in {pw_deg,pred_deg} and pw_deg != pred_deg:
            bonsai_likelihood_agreement = False
            warn_only = False
            unlikely_pair_info_list.append([focal_id,id1,id2,pw_deg,pred_deg])
        elif pw_deg[2] is None:
            if abs_pred_deg <= 5:
                bonsai_likelihood_agreement = False
                unlikely_pair_info_list.append([focal_id,id1,id2,pw_deg,pred_deg])
                if abs_pred_deg <= 3:
                    warn_only = False # if it's super unlikely, actually raise an exception rather than a warning.
        elif abs(abs_pw_deg - abs_pred_deg) > deg_diff:
            if abs_pred_deg == float('inf'):
                if abs_pw_deg <= 5:
                    bonsai_likelihood_agreement = False
                    unlikely_pair_info_list.append([focal_id,id1,id2,pw_deg,pred_deg])
            else:
                bonsai_likelihood_agreement = False
                warn_only = False
                unlikely_pair_info_list.append([focal_id,id1,id2,pw_deg,pred_deg])

    if bonsai_likelihood_agreement:
        return True
    else:
        err_message = ''
        for unlikely_pair_info in unlikely_pair_info_list:
            err_message += '\n' + "Unlikely pair. focal_id={}, id1={}, id1={}, pw_deg={}, pred_deg={}".format(*unlikely_pair_info)
        if throw_exception:
            if warn_only or user_error_detected:
                print(err_message)
            else:
                raise UnlikelyRelationshipException(err_message)
        else:
            return False

def enforce_rel_types( pairwise_log_likes , deg_set , small_log_prob ):
    """
    set all entries of pairwise_log_likes to small_log_prob if they are not in deg_set
    """
    
    for deg,log_like in iteritems(pairwise_log_likes):
        if deg not in deg_set:
            pairwise_log_likes[deg] = small_log_prob


def none_zero(val):
    if val is None:
        return 0
    else:
        return val
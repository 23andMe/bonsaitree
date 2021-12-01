from typing import Any, Dict, List, Set, Tuple, FrozenSet
import ast
import copy
import numbers
import numpy as np

from itertools import combinations

from .copytools import deepcopy
from .distributions import load_distributions, stringify_keys, unstringify_keys
from .point_predictor import get_distant_rel_log_like_by_ids, construct_point_prediction_group, point_predictions

class PedigreeObject(object):
    __slots__ = [
        'sex_dict', 'age_dict', 'use_age_info',
        'pedigree_log_likelihood', 'rels', 'up_pedigree_dict',
        'down_pedigree_dict', 'min_parent_ind', 'use_age_info',
        'pairwise_log_likelihoods','ibd_stats','rel_dict','all_ids',
        'genotyped_ids','ids_with_unresolvable_grandparents',
        'mispaired_parents_set','point_prediction_group',
        'distributions',
    ]

    sex_index_dict = {'M' : 1, 'F' : 0 , None : None }
    opposite_sex = { 'M' : 'F' , 'F' : 'M' , None : None }
    sex_list = ['F','M',None]

    def __deepcopy__(self, memo):
        cls = self.__class__
        newcopy = cls.__new__(cls)
        newcopy.use_age_info = self.use_age_info
        newcopy.sex_dict = self.sex_dict
        newcopy.age_dict = self.age_dict
        newcopy.min_parent_ind = self.min_parent_ind
        newcopy.pedigree_log_likelihood = self.pedigree_log_likelihood
        newcopy.pairwise_log_likelihoods = self.pairwise_log_likelihoods
        newcopy.ibd_stats = self.ibd_stats
        newcopy.ids_with_unresolvable_grandparents = self.ids_with_unresolvable_grandparents
        newcopy.mispaired_parents_set = self.mispaired_parents_set
        newcopy.point_prediction_group = self.point_prediction_group
        newcopy.distributions = self.distributions

        newcopy.up_pedigree_dict = deepcopy(self.up_pedigree_dict)
        newcopy.down_pedigree_dict = deepcopy(self.down_pedigree_dict)
        newcopy.rel_dict = deepcopy(self.rel_dict)
        newcopy.rels = deepcopy(self.rels)
        newcopy.all_ids = deepcopy(self.all_ids)
        newcopy.genotyped_ids = deepcopy(self.genotyped_ids)
        return newcopy

    def __init__(self, up_pedigree_dict=None, pairwise_log_likelihoods=None, ibd_stats=None, sex_dict=None, age_dict=None, use_age_info=True):

        self.use_age_info = use_age_info
        self.sex_dict = sex_dict
        if self.sex_dict is None:
            self.sex_dict = {}
        self.age_dict = age_dict
        if self.age_dict is None:
            self.age_dict = {}
        self.pairwise_log_likelihoods = pairwise_log_likelihoods # Dictionary of the form pairwise_log_likelihoods[ind_id][rel_id][(up_meioses,down_meioses,num_common_ancs)] = inferred log likelihood of this relationship.
        if self.pairwise_log_likelihoods is None:
            self.pairwise_log_likelihoods = {}
        self.ibd_stats = ibd_stats
        if self.ibd_stats is None:
            self.ibd_stats = {}
        up_pedigree_dict = self.remove_none_parents(up_pedigree_dict)
        self.up_pedigree_dict = up_pedigree_dict # Maps ids to parents
        if self.up_pedigree_dict is None:
            self.up_pedigree_dict = {}

        self.pedigree_log_likelihood = 0

        if self.ibd_stats:
            profile_information  = {i : {'age' : u[1], 'sex' : u[0]} for i,u in self.up_pedigree_dict.items() if self.is_genotyped(i)}
            for i,age in self.age_dict.items():
                if not self.is_genotyped(i):
                    continue
                sex = self.sex_dict.get(i,None)
                profile_information.update({i : {'age' : age, 'sex' : sex}})
        else:
            profile_information = {}
        #profile_information = {i : {'age' : self.age_dict[i], 'sex' : self.sex_dict[i]} for i in self.age_dict}
        self.distributions = load_distributions()
        self.point_prediction_group = construct_point_prediction_group(profile_information, self.ibd_stats)

        self.rels = dict() # Has the form dict[resid1][resid2] = np.array([up,down,num_ancs]).
        self.rel_dict = dict() # Has the form dict[resid] = { 'anc' : set_of_direct_ancestor_ids , 'desc' : set_of_direct_descendant_ids , 'rel' : set_of_all_other_relative_ids }
        self.all_ids = list() # Set of all IDs, genotyped or otherwise in the pedigree.
        self.genotyped_ids = set() # Set of all IDs that are part of a key in self.ibd_stats
        self.ids_with_unresolvable_grandparents = set() # Set of ids of individuals whose grandparental lineage placement cannot be resolved (no data)
        self.mispaired_parents_set = set() # set of individuals for whom the parental lineages are probably mispaired or have something else wrong because IBD segments shared between the descendants of the parents and the two parental relative clusters overlap in places. This can't happen without inbreeding or background IBD.

        self.extend_up_pedigree_dict() # Make entries for pedigree founders if they don't exist already.
        self.get_down_pedigree_dict_from_up_pedigree_dict() # Maps ids to [sex,age,kid1,kid2,...]

        self.update_all_ids()
        self.set_min_parent_ind()
        self.update_all_rels()
        self.set_genotyped_ids()
        self.compute_pedigree_log_likelihood()

    @classmethod
    def init_from_state(cls, state_dict):
        """
        Initialize the pedigree object from an existing state.
        """
        use_age_info = state_dict['use_age_info']
        sex_dict = unstringify_keys(state_dict['sex_dict'])
        age_dict = unstringify_keys(state_dict['age_dict'])
        pairwise_log_likelihoods = unstringify_keys(state_dict['pairwise_log_likelihoods']) # Dictionary of the form pairwise_log_likelihoods[ind_id][rel_id][(up_meioses,down_meioses,num_common_ancs)] = inferred log likelihood of this relationship.
        ibd_stats = unstringify_keys(state_dict['ibd_stats'])
        up_pedigree_dict = unstringify_keys(state_dict['up_pedigree_dict'])

        return cls(up_pedigree_dict=up_pedigree_dict,
                   pairwise_log_likelihoods=pairwise_log_likelihoods,
                   ibd_stats=ibd_stats,
                   sex_dict=sex_dict,
                   age_dict=age_dict,
                   use_age_info=use_age_info)

    def get_state_dict(self):
        """
        Get a json serializable dictionary storing the state of teh pedigre builder.
        This is enough info to initialize a new builder to the same state.
        """
        return {
            'use_age_info': self.use_age_info,
            'sex_dict': stringify_keys(self.sex_dict),
            'age_dict': stringify_keys(self.age_dict),
            'pairwise_log_likelihoods': stringify_keys(self.pairwise_log_likelihoods),
            'ibd_stats': stringify_keys(self.ibd_stats),
            'up_pedigree_dict': stringify_keys(self.up_pedigree_dict),
        }

    # MIN: Find the minimum of all elements in a vector, excluding nones.
    # If all elements are None, return None
    def _none_min(self, vec):
        try:
            return min(v for v in vec if v is not None)
        except:
            return None

    # MAX: Find the maximum of all elements in a vector, excluding nones.
    # If all elements are None, return None
    def _none_max(self, vec):
        try:
            return max(v for v in vec if v is not None)
        except:
            return None

    # Get a new index for for someone in the pedigree and update the lowest index.
    def get_new_ind(self):
        self.min_parent_ind -= 1
        return self.min_parent_ind

    # Check whether an ID has two genotyped parents in the pedigree
    # (parents are genotyped if *their* parents are specified... i.e.,
    # if they appear in the up_pedigree_dict)
    def is_orphan(self,resid):
        if resid in self.up_pedigree_dict:
            return False
        else:
            return True

    # Check whether an ID has children in the pedigree
    def is_leaf(self,resid):
        if resid in self.down_pedigree_dict:
            return False
        else:
            return True

    def is_founder(self,resid):
        if resid in self.down_pedigree_dict:
            if resid not in self.up_pedigree_dict:
                return True
            elif all(pid is None for pid in self.up_pedigree_dict[resid][2:]):
                return True
            else:
                return False
        elif (resid in self.up_pedigree_dict) and (len(self.up_pedigree_dict) == 1):
            return True
        else:
            return False

    def set_genotyped_ids(self):
        self.genotyped_ids = set()
        for pair_key in self.ibd_stats.keys():
            self.genotyped_ids |= pair_key

    def is_genotyped(self, resid):
        if resid > 0:
            return True
        else:
            return False

    def remove_none_parents(self,up_dict):
        new_up_dict = dict()
        for uid,info in up_dict.items():
            new_info = list(info[:2]) + list(filter(lambda x: x is not None,info[2:]))
            new_up_dict[uid] = new_info
        return new_up_dict

    def update_ungenotyped_inds(self , new_ind, debug=False):
        """
        Update ungenotyped indices.
        All updated indices will be strictly lower than new_ind
        Useful for merging pedigrees where one might have a lower min index.
        """

        self.set_min_parent_ind(new_ind,override=False)

        # Make a dictionary mapping old ungenotyped indices to new indices
        resid_update_dict = dict()
        for resid,info in self.down_pedigree_dict.items():
            if isinstance(resid,int) and resid < 0:
                self.get_new_ind()
                resid_update_dict[resid] = self.min_parent_ind

        new_up_dict = dict()
        for resid,info in self.up_pedigree_dict.items():
            if resid in resid_update_dict:
                new_id = resid_update_dict[resid]
            else:
                new_id = resid
            new_info = info[:2]
            for ind,parent_id in enumerate(info[2:]):
                if parent_id in resid_update_dict:
                    new_parent_id = resid_update_dict[parent_id]
                else:
                    new_parent_id = parent_id
                new_info.append(new_parent_id)
            new_up_dict[new_id] = new_info
        self.up_pedigree_dict = new_up_dict

        new_down_dict = dict()
        for resid,info in self.down_pedigree_dict.items():
            if resid in resid_update_dict:
                new_id = resid_update_dict[resid]
            else:
                new_id = resid
            new_info = info[:2]
            for ind,child_id in enumerate(info[2:]):
                if child_id in resid_update_dict:
                    new_child_id = resid_update_dict[child_id]
                else:
                    new_child_id = child_id
                new_info.append(new_child_id)
            new_down_dict[new_id] = new_info
        self.down_pedigree_dict = new_down_dict

        # Rename age and sex dict entries
        for old_id,new_id in resid_update_dict.items():
            if old_id in self.sex_dict:
                self.sex_dict[new_id] = self.sex_dict[old_id]
                del self.sex_dict[old_id]
            if old_id in self.age_dict:
                self.age_dict[new_id] = self.age_dict[old_id]
                del self.age_dict[old_id]

        # Rename pairwise likelihood dict entries
        pw_ll_outer_keys = self.pairwise_log_likelihoods.keys()
        for id1 in pw_ll_outer_keys:
            pw_ll_inner_keys = self.pairwise_log_likelihoods[id1].keys()
            for id2 in pw_ll_inner_keys:
                if id2 in resid_update_dict:
                    new_id = resid_update_dict[id2]
                    self.pairwise_log_likelihoods[id1][new_id] = self.pairwise_log_likelihoods[id1][id2]
                    del self.pairwise_log_likelihoods[id1][id2]
            if id1 in resid_update_dict:
                new_id = resid_update_dict[id1]
                self.pairwise_log_likelihoods[new_id] = self.pairwise_log_likelihoods[id1]
                del self.pairwise_log_likelihoods[id1]

        # Rename rels dict entries
        rels_outer_keys = [*self.rels]
        for id1 in rels_outer_keys:
            rels_inner_keys = [*self.rels[id1]]
            for id2 in rels_inner_keys:
                if id2 in resid_update_dict:
                    new_id = resid_update_dict[id2]
                    self.rels[id1][new_id] = self.rels[id1][id2]
                    del self.rels[id1][id2]
            if id1 in resid_update_dict:
                new_id = resid_update_dict[id1]
                self.rels[new_id] = self.rels[id1]
                del self.rels[id1]

        # Rename ibd_stats dict entries
        ibd_stats_keys = [*self.ibd_stats]
        for key in ibd_stats_keys:
            id1,id2 = list(key)
            if id1 in resid_update_dict and id2 in resid_update_dict:
                new_id1 = resid_update_dict[id1]
                new_id2 = resid_update_dict[id2]
                new_key = frozenset({new_id1,new_id2})
                self.ibd_stats[new_key] = self.ibd_stats[key]
                del self.ibd_stats[key]
            elif id1 in resid_update_dict:
                new_id1 = resid_update_dict[id1]
                new_key = frozenset({new_id1,id2})
                self.ibd_stats[new_key] = self.ibd_stats[key]
                del self.ibd_stats[key]
            elif id2 in resid_update_dict:
                new_id2 = resid_update_dict[id2]
                new_key = frozenset({id1,new_id2})
                self.ibd_stats[new_key] = self.ibd_stats[key]
                del self.ibd_stats[key]

        # Rename self.rel_dict entries
        rel_dict_keys = [*self.rel_dict]
        for key in rel_dict_keys:
            for rel_type in [*self.rel_dict[key]]:
                rels = self.rel_dict[key][rel_type]
                for rel in rels:
                    if rel in resid_update_dict:
                        new_id = resid_update_dict[rel]
                        ind = self.rel_dict[key][rel_type].add(new_id)
                        self.rel_dict[key][rel_type].remove(rel)
            if key in resid_update_dict:
                new_id = resid_update_dict[key]
                self.rel_dict[new_id] = self.rel_dict[key]
                del self.rel_dict[key]

        # Rename self.all_ids entries
        all_ids = deepcopy(self.all_ids)
        for old_id in all_ids:
            if old_id in resid_update_dict:
                new_id = resid_update_dict[old_id]
                ind = self.all_ids.index(old_id)
                self.all_ids[ind] = new_id

        self.update_all_rels()

        return resid_update_dict


    def get_rel_label( self , id1 , id2 ):

        deg_to_label = { (0,0,2) : ['YOU' , 0],
                         (0,0,2) : ['TWIN' , 1],
                         (1,0,1) : ['FATHER' , 2],
                         (0,1,1) : ['SON' , 4],
                         (1,1,2) : ['BROTHER' , 6],
                         (1,1,1) : ['HALF_BROTHER' , 8],
                         (2,0,1) : ['GRANDFATHER' , 10],
                         (0,2,1) : ['GRANDSON' , 12],
                         (2,1,2) : ['UNCLE' , 14],
                         (1,2,2) : ['NEPHEW' , 16],
                         (3,0,1) : ['GREAT_GRANDFATHER' , 18],
                         (0,3,1) : ['GREAT_GRANDSON' , 19],
                         (3,1,2) : ['GREAT_UNCLE' , 22],
                         (1,3,2) : ['GREAT_NEPHEW' , 24],
                         (2,2,2) : ['FIRST_COUSIN' , 26],
                         (2,3,2) : ['FIRST_COUSIN_ONCE_REMOVED' , 27],
                         (2,4,2) : ['FIRST_COUSIN_TWICE_REMOVED' , 28],
                         (3,3,2) : ['SECOND_COUSIN' , 29],
                         (3,4,2) : ['SECOND_COUSIN_ONCE_REMOVED' , 30],
                         (3,5,2) : ['SECOND_COUSIN_TWICE_REMOVED' , 31],
                         (4,4,2) : ['THIRD_COUSIN' , 32],
                         (4,5,2) : ['THIRD_COUSIN_ONCE_REMOVED' , 33],
                         (4,6,2) : ['THIRD_COUSIN_TWICE_REMOVED' , 34],
                         (5,5,2) : ['FOURTH_COUSIN' , 35],
                         (6,6,2) : ['FIFTH_COUSIN' , 38],
                         (7,7,2) : ['SIXTH_COUSIN' , 41],
                         (-float('inf'),-float('inf'),None) : ['DISTANT_COUSIN' , 44] }

        rel_deg = self.rels[id1][id2]
        if rel_deg[0] >= 1:
            if rel_deg in [(1,1,2),(1,1,1)]:
                side = None
            else:
                mom_id, dad_id = self.up_pedigree_dict[id1][2:4]
                mom_rel = self.rels[mom_id].get(id2)
                dad_rel = self.rels[dad_id].get(id2)
                if mom_rel[2] != None and dad_rel[2] != None:
                    side = 'Both'
                elif mom_rel[2] != None:
                    side = 'Maternal'
                elif dad_rel[2] != None:
                    side = 'Paternal'
        else:
            side = None

        label = deg_to_label[rel_deg]
        return label , side

    # We can input a pedigree dict that doesn't have up entries for founders. Make them.
    def extend_up_pedigree_dict(self):
        new_up_dict = dict()
        for resid,info in self.up_pedigree_dict.items():
            new_up_dict[resid] = info
            for parent_id in info[2:]:
                if parent_id not in self.up_pedigree_dict:
                    new_up_dict[parent_id] = [None,None]
        self.up_pedigree_dict = new_up_dict

    # Gets the down pedigree dict mapping parents to [sex,age,kid1,kid2,...] from self.up_pedigree_dict
    def get_down_pedigree_dict_from_up_pedigree_dict(self):
        self.down_pedigree_dict = {}
        for resid,info in self.up_pedigree_dict.items():
            for parent_id in info[2:]:
                if not parent_id in self.down_pedigree_dict: # Add parent
                    parent_info = self.up_pedigree_dict.get(parent_id)
                    if parent_info:
                        parent_sex = parent_info[0]
                        parent_age = parent_info[1]
                        self.down_pedigree_dict[parent_id] = [ parent_sex , parent_age ]
                    else:
                        self.down_pedigree_dict[parent_id] = [ None , None ]
                self.down_pedigree_dict[parent_id].append(resid)

    # When connecting a parent and a child, update the rel pairs formed by this connection
    def update_rels( self , child_id , parent_id ):

        # Initialize entries if they're empty
        if child_id not in self.rel_dict:
            self.rel_dict[child_id] = { 'anc' : set() , 'desc' : set() , 'rel' : set() }

        if parent_id not in self.rel_dict:
            self.rel_dict[parent_id] = { 'anc' : set() , 'desc' : set() , 'rel' : set() }

        if child_id not in self.rels:
            self.rels[child_id] = dict()
            self.rels[child_id][child_id] = (0,0,2)

        if parent_id not in self.rels:
            self.rels[parent_id] = dict()
            self.rels[parent_id][parent_id] = (0,0,2)

        # Connect child and descendants to parent and ancestors and non-descendant relatives of the parent
        for rel_id in self.rel_dict[parent_id]['anc'] | self.rel_dict[parent_id]['rel'] | set([parent_id]):
            par_rel_deg = self.rels[parent_id][rel_id]
            if rel_id == parent_id:
                par_rel_deg = (0,0,1)
            for desc_id in self.rel_dict[child_id]['desc'] | set([child_id]):
                diff = self.rels[desc_id][child_id][0] + 1 # Up degree from descendant to parent
                new_up_deg = ( par_rel_deg[0] + diff , par_rel_deg[1] , par_rel_deg[2] )
                new_down_deg = ( new_up_deg[1] , new_up_deg[0] , new_up_deg[2] )
                self.rels[desc_id][rel_id] = new_up_deg
                self.rels[rel_id][desc_id] = new_down_deg

        # Connect child and its descendants to other descendants of the parent
        other_parent_descendants = (self.rel_dict[parent_id]['desc'] - self.rel_dict[child_id]['desc']) - set([child_id]) # This is just for safety. Presumably parent and child are not attached so we could just use self.rel_dict[parent_id]['desc'].
        for par_desc_id in other_parent_descendants:
            for child_desc_id in self.rel_dict[child_id]['desc'] | set([child_id]):
                down_deg = self.rels[parent_id][par_desc_id][1]
                up_deg = self.rels[child_desc_id][parent_id][0]
                existing_deg = self.rels[child_desc_id].get(par_desc_id)
                if not existing_deg:
                    self.rels[child_desc_id][par_desc_id] = ( up_deg , down_deg , 1 )
                    self.rels[par_desc_id][child_desc_id] = ( down_deg , up_deg , 1 )
                else:
                    common_ancs = ( self.rel_dict[child_desc_id]['anc'] & self.rel_dict[par_desc_id]['anc'] ) - set([parent_id])
                    if not common_ancs: # The two were already joined by this parent
                        continue
                    else:
                        self.rels[child_desc_id][par_desc_id] = ( up_deg , down_deg , 2 )
                        self.rels[par_desc_id][child_desc_id] = ( down_deg , up_deg , 2 )

        # Update self.rel_dict
        self.rel_dict[child_id]['anc'] |= self.rel_dict[parent_id]['anc'] | set([parent_id])
        self.rel_dict[child_id]['rel'] |= self.rel_dict[parent_id]['rel']
        self.rel_dict[child_id]['rel'] |= (self.rel_dict[parent_id]['desc'] - self.rel_dict[child_id]['desc']) - set([child_id]) # All other descendants of parent_id who are not the child or their descenadants

        for child_desc_id in self.rel_dict[child_id]['desc']:
            self.rel_dict[child_desc_id]['anc'] |= self.rel_dict[child_id]['anc']
            self.rel_dict[child_desc_id]['rel'] |= self.rel_dict[child_id]['rel']

        for rel_id in self.rel_dict[child_id]['rel']:
            self.rel_dict[rel_id]['rel'] |= self.rel_dict[child_id]['desc'] | set([child_id])

        for anc_id in self.rel_dict[child_id]['anc']:
            self.rel_dict[anc_id]['desc'] |= self.rel_dict[child_id]['desc'] | set([child_id])


    def update_all_rels(self):

        self.rels = dict()

        for id1,id2 in combinations(self.all_ids, r=2):

            if id1 not in self.rels:
                self.rels[id1] = dict()
            if id2 not in self.rels:
                self.rels[id2] = dict()

            anc_dict1 = self.get_ancestor_dict2(id1,dict(),0)
            anc_dict2 = self.get_ancestor_dict2(id2,dict(),0)

            if id2 in anc_dict1:
                deg = anc_dict1[id2]
                self.rels[id1][id2] = (deg,0,1)
                self.rels[id2][id1] = (0,deg,1)
            elif id1 in anc_dict2:
                deg = anc_dict2[id1]
                self.rels[id1][id2] = (0,deg,1)
                self.rels[id2][id1] = (deg,0,1)
            elif not set(anc_dict1.keys()) & set(anc_dict2.keys()):
                self.rels[id1][id2] = (float('inf'),float('inf'),None)
                self.rels[id2][id1] = (float('inf'),float('inf'),None)
            else:
                common_ancs = set(anc_dict1.keys()) & set(anc_dict2.keys())
                min_deg1 = float('inf')
                min_deg2 = float('inf')
                common_anc_set = set()
                for anc_id in common_ancs:
                    deg1 = anc_dict1[anc_id]
                    deg2 = anc_dict2[anc_id]
                    if deg1 < min_deg1 and deg2 < min_deg2:
                        min_deg1 = deg1
                        min_deg2 = deg2
                        common_anc_set = set([anc_id])
                    elif deg1 == min_deg1 and deg2 == min_deg2:
                        common_anc_set.add(anc_id)

                num_ancs = len(common_anc_set)
                self.rels[id1][id2] = (min_deg1,min_deg2,num_ancs)
                self.rels[id2][id1] = (min_deg2,min_deg1,num_ancs)

        self.rel_dict = dict()
        for uid in self.rels.keys():
            if uid not in self.rel_dict:
                self.rel_dict[uid] = {'anc' : set(), 'desc' : set(), 'rel' : set()}
            for rel_id,rel_deg in self.rels[uid].items():
                if (rel_deg[0] == 0) and (rel_deg[1] > 0):
                    self.rel_dict[uid]['desc'].add(rel_id)
                elif (rel_deg[0] > 0) and (rel_deg[1] == 0):
                    self.rel_dict[uid]['anc'].add(rel_id)
                elif (rel_deg[0] > 0) and (rel_deg[1] > 0) and (sum(rel_deg[:2]) < float('inf')):
                    self.rel_dict[uid]['rel'].add(rel_id)

        # When the pedigree only has one person, then we haven't made a rel_dict entry for them yet.
        for uid in self.all_ids:
            if uid not in self.rels:
                self.rel_dict[uid] = {'anc' : set(), 'desc' : set(), 'rel' : set()}

        for id1,id2 in combinations(self.all_ids, r=2):
            try:
                deg = self.rels[id1][id2]
            except:
                self.rels[id1][id2] = (float('inf'),float('inf'),None)
                self.rels[id2][id1] = (float('inf'),float('inf'),None)

        for uid in self.all_ids:
            if uid not in self.rels: # If it's a pedigree with just one person, self.rels[uid] might not be initialized.
                self.rels[uid] = dict()
            self.rels[uid][uid] = (0,0,2)


    def get_ancestor_dict2(self,ind_id,anc_dict=dict(),count=0):
        anc_dict[ind_id] = count
        parent_ids = self.up_pedigree_dict.get(ind_id,[])[2:]
        if len(parent_ids) == 0:
            return anc_dict
        for pid in parent_ids:
            temp_dict = self.get_ancestor_dict2(pid,anc_dict,count+1)
            for anc_id,deg in temp_dict.items():
                if anc_id not in anc_dict:
                    anc_dict[anc_id] = deg
        return anc_dict


    def update_all_ids(self):
        self.all_ids = list( set( self.up_pedigree_dict.keys() ) | set( self.down_pedigree_dict.keys() ) )


    def set_min_parent_ind(self, new_ind=None, override=False):
        if new_ind is None: # Find the min parent ind.
            self.update_all_ids()
            self.set_genotyped_ids()
            ungenotyped_id_set = set(self.all_ids) - self.genotyped_ids
            self.min_parent_ind = 0
            for subj_id in ungenotyped_id_set:
                if isinstance(subj_id, numbers.Number) and subj_id < self.min_parent_ind:
                    self.min_parent_ind = subj_id
        elif new_ind >= self.min_parent_ind and override == False:
            raise Exception("new_ind is greater than min_parent_ind.")
        else:
            self.min_parent_ind = new_ind

    def update_log_likelihood(self , new_id):
        if not self.is_genotyped(new_id):
            return
        for rel_id in set(self.all_ids) - set([new_id]) :
            if self.is_genotyped(rel_id):
                rel_deg = self.rels[new_id].get(rel_id, (float('inf') , float('inf') , None) ) # Return unrelated degree if not related
                log_like = self.pairwise_log_likelihoods[new_id][rel_id].get(rel_deg)
                if log_like:
                    self.pedigree_log_likelihood += log_like
                else:
                    log_like = get_distant_rel_log_like_by_ids(
                        unique_id_1 = new_id,
                        unique_id_2 = rel_id,
                        up_meioses = rel_deg[0],
                        down_meioses = rel_deg[1],
                        num_ancestors = rel_deg[2],
                        point_prediction_group = self.point_prediction_group,
                        distribution_models = self.distributions,
                    )
                    if isinstance(log_like,list) or isinstance(log_like,np.ndarray):
                        log_like = log_like[0]
                    self.pedigree_log_likelihood += log_like

    def compute_pedigree_log_likelihood(self):
        self.pedigree_log_likelihood = 0
        if not self.pairwise_log_likelihoods:
            return
        gt_id_set = {uid for uid in self.up_pedigree_dict.keys() if uid > 0}
        num_genotyped = len(gt_id_set)
        if num_genotyped == 1:
            self.pedigree_log_likelihood = 0
        else:
            for id1,id2 in combinations(gt_id_set, r=2):
                rel_deg = self.rels[id1].get(id2, (float('inf') , float('inf') , None) ) # Return unrelated degree if not related
                log_like = self.pairwise_log_likelihoods[id1][id2].get(rel_deg)
                if log_like:
                    self.pedigree_log_likelihood += log_like
                else:
                    log_like = get_distant_rel_log_like_by_ids(
                        unique_id_1 = id1,
                        unique_id_2 = id2,
                        up_meioses = rel_deg[0],
                        down_meioses = rel_deg[1],
                        num_ancestors = rel_deg[2],
                        point_prediction_group = self.point_prediction_group,
                        distribution_models = self.distributions,
                    )
                    if isinstance(log_like,list) or isinstance(log_like,np.ndarray):
                        log_like = log_like[0]
                    self.pedigree_log_likelihood += log_like


    def replace_individual(self, old_id, new_id, new_sex, new_age):
        """
        Replace an existing individual in the pedigree with a new individual
        and update the pedigree accordingly.
        """

        up_entry = None
        if old_id in self.up_pedigree_dict:
            up_entry = self.up_pedigree_dict[old_id]
            old_sex = up_entry[0]
            old_age = up_entry[1]
        down_entry = None
        if old_id in self.down_pedigree_dict:
            down_entry = self.down_pedigree_dict[old_id]
            old_sex = down_entry[0]
            old_age = down_entry[1]
        if (old_id not in self.up_pedigree_dict) and (old_id not in self.down_pedigree_dict):
            return False # id must be in the pedigree

        if new_sex is None:
            new_sex = old_sex
        if new_age is None:
            new_age = old_age

        # check that new_age is consistent with parent and child ages of old_id
        if self.use_age_info:
            if old_id in self.up_pedigree_dict:
                old_parent_ages = [self.down_pedigree_dict[parent_id][1] for parent_id in self.up_pedigree_dict[old_id][2:]]
                min_old_parent_age = self._none_min(old_parent_ages)
                if min_old_parent_age is not None and new_age is not None:
                    if not new_age <= min_old_parent_age:
                        return False
            if old_id in self.down_pedigree_dict:
                old_kid_ages = [self.up_pedigree_dict[child_id][1] for child_id in self.down_pedigree_dict[old_id][2:]]
                max_old_kid_age = self._none_max(old_kid_ages)
                if max_old_kid_age is not None and new_age is not None:
                    if not new_age >= max_old_kid_age:
                        return False

        # Replace old_id with new_id everywhere in the up and down pedigree dicts.
        if up_entry:
            del self.up_pedigree_dict[old_id]
            self.up_pedigree_dict[new_id] = [new_sex,new_age] + up_entry[2:]
            for parent in up_entry[2:]:
                kid_index = self.down_pedigree_dict[parent][2:].index(old_id)
                self.down_pedigree_dict[parent][2+kid_index] = new_id
        if down_entry:
            del self.down_pedigree_dict[old_id]
            self.down_pedigree_dict[new_id] = [new_sex,new_age] + down_entry[2:]
            for child_id in down_entry[2:]:
                parent_index = self.up_pedigree_dict[child_id][2:].index(old_id)
                self.up_pedigree_dict[child_id][2+parent_index] = new_id

        # Replace old_id etnry in self.rel_dict
        info = self.rel_dict[old_id]
        self.rel_dict[new_id] = info
        del self.rel_dict[old_id]

        for desc_id in self.rel_dict[new_id]['desc']:
            self.rel_dict[desc_id]['anc'].remove(old_id)
            self.rel_dict[desc_id]['anc'].add(new_id)

        for anc_id in self.rel_dict[new_id]['anc']:
            self.rel_dict[anc_id]['desc'].remove(old_id)
            self.rel_dict[anc_id]['desc'].add(new_id)

        for rel_id in self.rel_dict[new_id]['rel']:
            self.rel_dict[rel_id]['rel'].remove(old_id)
            self.rel_dict[rel_id]['rel'].add(new_id)

        # Replace old_id etnry in self.rels
        info = self.rels[old_id]
        self.rels[new_id] = info
        del self.rels[old_id]

        info = self.rels[new_id][old_id]
        self.rels[new_id][new_id] = info
        del self.rels[new_id][old_id]

        other_rel_ids = self.rel_dict[new_id]['anc'] |  self.rel_dict[new_id]['desc'] | self.rel_dict[new_id]['rel']
        for rel_id in other_rel_ids:
            info = self.rels[rel_id][old_id]
            self.rels[rel_id][new_id] = info
            del self.rels[rel_id][old_id]

        self.all_ids.append(new_id)
        self.all_ids.remove(old_id)

        if self.is_genotyped(new_id):
            self.update_log_likelihood(new_id)

        self.set_min_parent_ind()

        return True


    def new_individual(self, new_id, sex, age, mom_id=None, dad_id=None):
        """
        Add a new individual and update the pedigree accordingly.
        """
        self.up_pedigree_dict[new_id] = [ sex , age ]
        self.rels[new_id] = { new_id : (0,0,2) }
        self.rel_dict[new_id] = { 'anc' : set() , 'desc' : set() , 'rel' : set() }

        if mom_id == True:
            mom_id = self.min_parent_ind-1
            self.min_parent_ind -= 1
        if dad_id == True:
            dad_id = self.min_parent_ind-1
            self.min_parent_ind -= 1

        if mom_id:
            if mom_id in self.up_pedigree_dict:
                mom_current_sex = self.up_pedigree_dict[mom_id][0]
                if mom_current_sex and mom_current_sex != 'F':
                    return False
            self.up_pedigree_dict[new_id].append(mom_id)
            if mom_id in self.down_pedigree_dict:
                self.down_pedigree_dict[mom_id].append(new_id)
            else:
                self.down_pedigree_dict[mom_id] = [ 'F' , None , new_id]

            if mom_id not in self.rels:
                self.rels[mom_id] = { mom_id : (0,0,2) }

            self.rels[new_id][mom_id] = (1,0,1)
            self.rels[mom_id][new_id] = (0,1,1)

            if mom_id not in self.rel_dict:
                self.rel_dict[mom_id] = { 'anc' : set() , 'desc' : set() , 'rel' : set() }

            self.rel_dict[new_id]['anc'].add(mom_id)
            self.rel_dict[mom_id]['desc'].add(new_id)

        if dad_id:
            if dad_id in self.up_pedigree_dict:
                mom_current_sex = self.up_pedigree_dict[dad_id][0]
                if mom_current_sex and mom_current_sex != 'F':
                    return False
            self.up_pedigree_dict[new_id].append(dad_id)
            if dad_id in self.down_pedigree_dict:
                self.down_pedigree_dict[dad_id].append(new_id)
            else:
                self.down_pedigree_dict[dad_id] = [ 'F' , None , new_id]

            if dad_id not in self.rels:
                self.rels[dad_id] = { dad_id : (0,0,2) }

            self.rels[new_id][dad_id] = (1,0,1)
            self.rels[dad_id][new_id] = (0,1,1)

            if dad_id not in self.rel_dict:
                self.rel_dict[dad_id] = { 'anc' : set() , 'desc' : set() , 'rel' : set() }

            self.rel_dict[new_id]['anc'].add(dad_id)
            self.rel_dict[dad_id]['desc'].add(new_id)

        self.all_ids.append(new_id)

    # Connect a parent and child.
    def connect_parent_child( self , child_id , parent_id):
        parents = self.up_pedigree_dict[child_id][2:]
        if parent_id not in parents and len(parents) < 2:
            self.up_pedigree_dict[child_id].append(parent_id)
        if parent_id not in self.down_pedigree_dict:
            parent_info = self.up_pedigree_dict[parent_id]
            parent_sex = parent_info[0]
            parent_age = parent_info[1]
            self.down_pedigree_dict[parent_id] = [ parent_sex , parent_age ]
        children = self.down_pedigree_dict[parent_id][2:]
        if child_id not in children:
            self.down_pedigree_dict[parent_id].append(child_id)
        self.update_rels( child_id=child_id , parent_id=parent_id)

    # Add a parent to a child
    def add_parent_for_child(self, child_id , parent_sex=None , parent_age=None , parent_id=None ):
        child_info = self.up_pedigree_dict[child_id]
        parents = child_info[2:]
        if len(parents) >= 2:
            raise Exception("Too many parents.")
        if len(parents) == 1:
            existing_parent_id = parents[0]
            existing_parent_info = self.down_pedigree_dict[existing_parent_id]
            existing_parent_sex = existing_parent_info[0]
            new_parent_sex = self.opposite_sex[existing_parent_sex]
            if parent_sex and parent_sex != new_parent_sex:
                raise Exception("New parent sex conflicts with existing parent sex.")
            elif parent_sex is None:
                parent_sex = new_parent_sex
        if self.use_age_info:
            child_age = child_info[1]
            if parent_age and parent_age <= child_age:
                raise Exception("Child age exceeds parent age.")
        if not parent_id:
            parent_id = self.get_new_ind()
        self.new_individual( new_id=parent_id , sex=parent_sex , age=parent_age )
        self.connect_parent_child( child_id=child_id , parent_id=parent_id)
        if self.is_genotyped(parent_id):
            self.update_log_likelihood(parent_id)
        return parent_id

    # Add a child to a parent
    def add_child_for_parent(self, parent_id , child_id=None , child_sex=None , child_age=None ):
        if parent_id in self.down_pedigree_dict:
            parent_info = self.down_pedigree_dict[parent_id]
        else:
            parent_info = self.up_pedigree_dict[parent_id][:2]
        if self.use_age_info:
            parent_age = parent_info[1]
            if child_age and parent_age and parent_age <= child_age:
                raise Exception("Child age exceeds parent age.")
        if not child_id:
            child_id = self.get_new_ind()
        self.new_individual( new_id=child_id , sex=child_sex , age=child_age )
        self.connect_parent_child( child_id=child_id , parent_id=parent_id)
        if self.is_genotyped(child_id):
            self.update_log_likelihood(child_id)
        return child_id

    def get_descendant_dict(self , resid):
        if (len(self.all_ids) == 1) and (resid == self.all_ids[0]):
            return { resid : 0 }
        elif resid not in self.all_ids:
            raise Exception("ID " + str(resid) + " not in pedigree.")
        else:
            return { desc_id : self.rels[desc_id][resid][0] for desc_id in self.rel_dict[resid]['desc'] | set([resid])}

    def get_ancestor_dict(self, resid):
        if (len(self.all_ids) == 1) and (resid == self.all_ids[0]):
            return { resid : 0 }
        elif resid not in self.all_ids:
            raise Exception("ID " + str(resid) + " not in pedigree.")
        else:
            return { anc_id : self.rels[resid][anc_id][0] for anc_id in self.rel_dict[resid]['anc'] | set([resid])}

    # Dictionary of the form dict[common_anc_id][desc_id] = deg for desc_id in resid_list
    def get_common_ancestor_dict(self, resid_list,get_mrcas=False):
        # Get a list of common anc dicts
        anc_dicts = {}
        for resid in resid_list:
            anc_dict = self.get_ancestor_dict(resid)
            anc_dicts[resid] = anc_dict
        # Find all common ancestors, if any
        init_resid = resid_list[0]
        common_ancs = set(anc_dicts[init_resid].keys())
        for resid,anc_dict in anc_dicts.items():
            common_ancs &= set(anc_dict.keys())
        # If no common ancs, return empty dict
        if len(common_ancs) == 0:
            return {}
        #Subset to only the most recent common ancestors
        if get_mrcas:
            mrcas = set(common_ancs)
            for ca in common_ancs:
                anc_dict = self.get_ancestor_dict(ca)
                anc_id_set = set(anc_dict.keys())
                anc_id_set.remove(ca)
                for anc_id in anc_id_set:
                    try:
                        mrcas.remove(anc_id)
                    except:
                        pass
            common_ancs = mrcas
        #Build the common anc dict
        common_anc_dict = { anc_id : {} for anc_id in common_ancs }
        for resid,anc_dict in anc_dicts.items():
            for anc_id in common_ancs:
                deg = anc_dict[anc_id]
                common_anc_dict[anc_id][resid] = deg
        # Return common anc dict
        return common_anc_dict


    def get_covering_ancestor_set(self, resid_set):
        """
        Find a set of ancestors for a set of genotyped individuals.
        This is a set of ancestors for which the union of their descendants
        is a (possibly proper) superset of resid_set. This is not exactly
        the minimal set of ancestors, but I believe it would be minimal
        if we threw out one member of each remaining spouse pair in the
        resulting set.

        Args:
            resid_set : set of ids for which to find the set of covering
                        ancestors
        """

        # get all ancestor ids for people in resid_set
        all_anc_id_set = set()
        for iid in resid_set:
            all_anc_id_set |= self.rel_dict[iid]['anc']

        # look over all pairs of ancestors. If any ancestor is the ancestor of
        # another ancestor, ask whether it covers more individuals. If not,
        # mark it for deletion by adding it to del_set.
        del_set = set()
        for id1,id2 in combinations(all_anc_id_set, r=2):
            anc_id = None
            desc_id = None
            if id1 in self.rel_dict[id2]['anc']:
                anc_id = id1
                desc_id = id2
            elif id2 in self.rel_dict[id1]['anc']:
                anc_id = id2
                desc_id = id1
            else:
                continue
            anc_desc_set = self.rel_dict[anc_id]['desc']
            desc_desc_set = self.rel_dict[desc_id]['desc']
            if len(anc_desc_set) <= len(desc_desc_set):
                del_set.add(anc_id)
            elif len(desc_desc_set) < len(anc_desc_set):
                del_set.add(desc_id)

        return all_anc_id_set - del_set


    def get_connecting_path_set(self,id1,id2):
        """
        Find all individuals on the path of ancestors from id1 to id2.
        Useful for finding all people who carry a shared IBD segment.
        """

        # get a dict of the form ca_dict[anc][desc] = deg
        common_anc_dict = self.get_common_ancestor_dict(resid_list=[id1,id2], get_mrcas=True)

        if len(common_anc_dict) == 0:
            return set()

        id1_anc_set = self.rel_dict[id1]['anc']
        id2_anc_set = self.rel_dict[id2]['anc']

        # There should only be two common ancestors who are a couple. Pick one.
        common_anc_id = list(common_anc_dict.keys())[0]
        anc_desc_dict = self.get_descendant_dict(common_anc_id)
        anc_desc_set = set(anc_desc_dict.keys())

        path_set_1 = id1_anc_set & anc_desc_set
        path_set_2 = id2_anc_set & anc_desc_set

        path_set = path_set_1 | path_set_2 # Set of individuals on the path between id1 and id2

        if id2 in self.rel_dict[id1]['anc']:
            path_set = path_set_1 - path_set_2
        elif id1 in self.rel_dict[id2]['anc']:
            path_set = path_set_2 - path_set_1

        return path_set


    def remove_node(self,node):
        """
        Remove a node and then delete all its references from
        up_pedigree_dict, down_pedigree_dict, rel_dict, rels,
        and all_ids.
        """

        # Remove node from the child lists of its parents
        if node in self.up_pedigree_dict:
            pids = self.up_pedigree_dict[node][2:]
            for pid in pids:
                pid_info = self.down_pedigree_dict[pid]
                pid_info = pid_info[:2] + [uid for uid in pid_info[2:] if uid != node] 
                self.down_pedigree_dict[pid] = pid_info
        try:
            del self.up_pedigree_dict[node]
        except:
            pass

        # Remove node from the parent lists of its children
        if node in self.down_pedigree_dict:
            cids = self.down_pedigree_dict[node][2:]
            for cid in cids:
                cid_info = self.up_pedigree_dict[cid]
                cid_info = cid_info[:2] + [uid for uid in cid_info[2:] if uid != node]
                self.up_pedigree_dict[cid] = cid_info
        try:
            del self.down_pedigree_dict[node]
        except:
            pass

        try:
            del self.rels[node]
        except:
            pass
        for other_node in self.rels.keys():
            if node in self.rels[other_node]:
                del self.rels[other_node][node]

        try:
            del self.rel_dict[node]
        except:
            pass
        for other_node,other_node_rel_dict in self.rel_dict.items():
            if node in other_node_rel_dict['anc']:
                other_node_rel_dict['anc'].remove(node)
            if node in other_node_rel_dict['rel']:
                other_node_rel_dict['rel'].remove(node)
            if node in other_node_rel_dict['desc']:
                other_node_rel_dict['desc'].remove(node)
            self.rel_dict[other_node] = other_node_rel_dict

        self.update_all_ids()


    def keep_nodes(self,keep_gt_node_set,include_parents=False):
        """
        Remove all nodes in node_list from up_pedigree_dict, down_pedigree_dict,
        rel_dict, rels, and all_ids. Also reset min_parent_ind. Also remove 
        any nodes that are no longer the MRCA of two genotyped nodes.
        """

        # Get all nodes that connect two genotyped people. Also record genotyped people
        keep_node_set = set()
        keep_node_set |= keep_gt_node_set
        for node1,node2 in combinations(keep_gt_node_set, r=2):
            keep_node_set.add(node1)
            keep_node_set.add(node2)
            path_set = self.get_connecting_path_set(node1,node2) | set(self.get_common_ancestor_dict(resid_list=[node1,node2], get_mrcas=True).keys()) # Need to make sure we get both common ancestors, not just one, as get_connecting_path_set chooses just one.
            keep_node_set |= path_set
            if include_parents:
                node1_pid_set = set(self.up_pedigree_dict[node1][2:])
                node2_pid_set = set(self.up_pedigree_dict[node2][2:])
                keep_node_set |= node1_pid_set
                keep_node_set |= node2_pid_set
                for node in path_set:
                    if node in self.up_pedigree_dict:
                        node_pid_set = set(self.up_pedigree_dict[node][2:])
                        if keep_node_set and node_pid_set: # if at least one parent is in the keep set, keep the other too
                            keep_node_set |= node_pid_set

        remove_set = set(self.all_ids) - keep_node_set

        for node in remove_set:
            self.remove_node(node)


    def get_independent_inds(self, ind_ids):
        """
        Return a list of ids of people, none of whom is descended from
        the other, and who are the most ancestral
        Args:
           ind_ids: identifiers of people in the pedigree
        Returns:
           indep_ind_list: if any pair of ids in ind_ids contains a direct
                           ancestor-child pair, only the oldest is reatained
                           in indep_ind_list.
        """
        ind_ids_set = set(ind_ids)
        indep_ind_set = set()
        for ind_id in ind_ids:
            ancestors = self.get_ancestor_dict(ind_id)
            del ancestors[ind_id]
            if not set(ancestors.keys()) & ind_ids_set:
                indep_ind_set.add(ind_id)
        return indep_ind_set


    # Set the sex of someone in the pedigree object
    def set_sex(self , ind_id , sex ):
        if ind_id in self.down_pedigree_dict:
            info = self.down_pedigree_dict[ind_id]
            existing_sex = info[0]
            if existing_sex and existing_sex != sex:
                raise Exception("Old sex does not match new sex.")
            children = info[2:]
            # Make sure sex is opposite of all spouse sexes. Otherwise, raise an exception.
            for child_id in children:
                parents = self.up_pedigree_dict[child_id][2:]
                other_parent_list = list( set(parents) - set([ind_id]) ) # A list with at most one entry.
                if other_parent_list:
                    other_parent_id = other_parent_list[0]
                    other_parent_sex = self.down_pedigree_dict[other_parent_id][0]
                    if sex and other_parent_sex and sex != self.opposite_sex[ other_parent_sex ]:
                        raise Exception("Sex conflicts with existing spousal sex.")
                    elif (sex is None) and other_parent_sex:
                        sex = self.opposite_sex[other_parent_sex]
            if sex and (existing_sex is None):
                self.down_pedigree_dict[ind_id][0] = sex
        if ind_id in self.up_pedigree_dict:
            existing_sex = self.up_pedigree_dict[ind_id][0]
            if sex and existing_sex and existing_sex != sex:
                raise Exception("Old sex does not match new sex.")
            elif sex and (existing_sex is None):
                self.up_pedigree_dict[ind_id][0] = sex

    # If the sex for someone is known, we can update the sex of their spouse.
    def update_sexes(self , resid=None):
        if resid is None:
            iter_list = [[resid, info] for resid,info in self.down_pedigree_dict.items()]
        elif resid in self.down_pedigree_dict:
            info = self.down_pedigree_dict[resid]
            iter_list = [[resid,info]]
        else:
            return
        for resid,info in iter_list:
            children = info[2:]
            all_parents = []
            for child_id in children:
                parents = self.up_pedigree_dict[child_id][2:]
                all_parents.extend(parents)
            partners = set(all_parents) - set([resid])
            partner_sexes = set()
            for partner_id in list(partners):
                partner_info = self.down_pedigree_dict[partner_id]
                partner_sex = partner_info[0]
                if partner_sex:
                    partner_sexes.add(partner_sex)
            if len(partner_sexes) > 1:
                raise Exception("Inconsistent sexes: " + str(resid))
            if len(partner_sexes) > 0:
                other_sex = list(partner_sexes)[0]
            else:
                other_sex = None
            sex = info[0]
            if sex and other_sex and (sex != self.opposite_sex[other_sex]):
                raise Exception("Inconsistent sexes: " + str(resid))
            if (other_sex is not None) and (sex is None):
                sex = self.opposite_sex[other_sex]
                self.set_sex( resid , sex )
            other_sex = self.opposite_sex[ sex ]
            for partner_id in list(partners):
                partner_info = self.down_pedigree_dict[partner_id]
                partner_sex = partner_info[0]
                if other_sex and partner_sex is None:
                    self.set_sex( partner_id , other_sex )
                    self.update_sexes( partner_id )

    #  True if any partner pair has conflicting sexes
    def inconsistent_sexes(self):
        inconsistent_sexes = False
        inconsistent_id = None
        for node_id,info in self.down_pedigree_dict.items():
            node_sex = info[0]
            partner_id_set = self.get_partner_id_set(node_id)
            partner_sex_set = {self.down_pedigree_dict[partner_id][0] for partner_id in partner_id_set}
            partner_sex_set -= {None}
            if (node_sex in partner_sex_set) or (len(partner_sex_set) > 1):
                inconsistent_sexes = True
                inconsistent_id = node_id
                break
        return inconsistent_sexes,inconsistent_id

    # get set of all partners of node_id
    def get_partner_id_set(self,node_id):
        child_id_list = self.down_pedigree_dict[node_id][2:]
        partner_id_set = set()
        for child_id in child_id_list:
            parent_id_list = self.up_pedigree_dict[child_id][2:]
            partner_id_set |= set(parent_id_list)
        return partner_id_set - {node_id}

    # Remove founders from the up pedigree_dict
    def strip_founders(self):
        copy_up_dict = deepcopy(self.up_pedigree_dict)
        for resid,info in copy_up_dict.items():
            if len(info) == 2:
                del self.up_pedigree_dict[resid]

    # Fill in missing parents for everyone in the up dict
    def fill_in_parents(self):
        copy_up_dict = deepcopy(self.up_pedigree_dict)
        for resid,info in copy_up_dict.items():
            num_missing_parents = 4 - len(info)
            if num_missing_parents >= 2:
                continue
            for ind in range(num_missing_parents):
                self.add_parent_for_child(resid)
        self.update_sexes()

    # Arbitrarily (but consistently) assign sexes to people whose sex is None
    def assign_sexes(self):
        for resid in self.down_pedigree_dict.keys():
            info = self.down_pedigree_dict[resid]
            sex = info[0]
            if sex is None:
                rand_sex = np.random.choice(['M','F'])
                try:
                    self.set_sex( resid , rand_sex )
                except:
                    self.set_sex( resid , self.opposite_sex[rand_sex] )
                self.update_sexes( resid )

    # Order parent sexes female first. If both parent sexes are None, do nothing.
    def order_sexes(self):
        copy_up_dict = deepcopy(self.up_pedigree_dict)
        for resid,info in copy_up_dict.items():
            if len(info) < 4:
                continue
            id1 = info[2]
            id1_sex = self.down_pedigree_dict[id1][0]
            if id1_sex == 'M':
                self.up_pedigree_dict[resid][2] = info[3]
                self.up_pedigree_dict[resid][3] = info[2]


    def adjust_sides(self, focal_id, ibd_segs, overlap_length_delta=2e6, flag_only=False):
        """
        Examine the clusters of relatives coming up off of each of
        the four grandparents of an individual. Two clusters are unlikely
        to be from grandparents who are spouses if one or more ibd
        segments from the two clusters overlaps substantially. This is
        because recombinations in the parent imply that segments are
        disjoint unless there was recent endogamy. Search for overlapping
        segments and, if found, rearrange the pedigree so that the ancestral
        relatives are assigned to ancestral lineages that do not branch off
        the two individuals in a spouse pair. Check first to see if
        descendants come off a spouse pair who can serve to triangulate
        without needing to check for overlapping segments (triangulating
        relatives trump overlaps). Recurse on the parents of focal_id.

        Note: we tried first doing this by looking for places where IBD
              segments with one cluster were spliced with IBD segments
              with a second cluster. This worked fine for exact IBD data,
              but it didn't work for inferred IBD. It's possible
              that this statistic would work okay if we have better IBD
              estimates.

        # Args
        focal_id:
            int()|str()
        ibd_segs:
            [[int(gt_id_1), int(gt_id_2), str(chromosome), int(start), int(end), bool(is_full_ibd), float(max_seg_cm)], ...]
        overlap_length_delta:
            int() : number of bases of overlap that must differ between the current overlap and the minimum overlap for us to believe there is enough evidence that the current lineage placement is wrong.
        """

        def tally_overlapping_segs(seg_list1, seg_list2):
            """
            Count the total length of segments in seg_list1 that overlap a segment in seg_list2.
            """
            total_overlap_length = 0
            for segA in seg_list1:
                for segB in seg_list2:
                    if segA[2] != segB[2]: # check whether the two segments are on the same chromosome. If not, continue
                        continue
                    if ((segB[4] >= segA[3]) and (segB[3] <= segA[4])) or ((segA[4] >= segB[3]) and (segA[3] <= segB[4])): # if the two segments overlap
                        total_overlap_length += min(segA[4],segB[4]) - max(segA[3],segB[3])
            return total_overlap_length

        if focal_id not in self.up_pedigree_dict:
            return
        elif len(self.up_pedigree_dict[focal_id][2:]) == 0:
            return
        elif len(self.up_pedigree_dict[focal_id][2:]) == 1: # only one parent.
            parent_id = self.up_pedigree_dict[focal_id][2]
            if len(self.up_pedigree_dict.get(parent_id, [])) == 4: # parent_id is in up_pedigree_dict and they have two parents. So flag them if their two parental clusters have overlapping IBD segments.
                gp1,gp2 = self.up_pedigree_dict[parent_id][2:]
                gp1_rels = {rel_id for rel_id in self.rel_dict.get(gp1,{'rel' : set()})['rel'] if self.is_genotyped(rel_id)}
                gp2_rels = {rel_id for rel_id in self.rel_dict.get(gp2,{'rel' : set()})['rel'] if self.is_genotyped(rel_id)}
                gp1_desc_set = set(self.get_descendant_dict(gp1).keys())
                gp2_desc_set = set(self.get_descendant_dict(gp2).keys())
                if gp1_desc_set == gp2_desc_set: # if they're equal then there are no offsrping that can be used for triangulation so we want to check for overlapping segments. Otherwise, skip them.
                    gp1gp2_desc_set = {uid for uid in gp1_desc_set if self.is_genotyped(uid)}
                    indep_desc_set = set(self.get_independent_inds(gp1gp2_desc_set))
                    gp1_segs = [seg for seg in ibd_segs if (seg[0] in indep_desc_set and seg[1] in gp1_rels) or (seg[0] in gp1_rels and seg[1] == indep_desc_set)] # segments shared between gp1 and their relatives
                    gp2_segs = [seg for seg in ibd_segs if (seg[0] in indep_desc_set and seg[1] in gp2_rels) or (seg[0] in gp2_rels and seg[1] in indep_desc_set)]
                    total_overlap_length = tally_overlapping_segs(gp1_segs, gp2_segs)
                    if total_overlap_length >= overlap_length_delta: # then one grandparent should probably have been placed on the other parent
                        if flag_only:
                            self.mispaired_parents_set.add(parent_id)
                        else:
                            new_parent_id = self.add_parent_for_child(focal_id)
                            self.up_pedigree_dict[parent_id] = self.up_pedigree_dict[parent_id][:2] + [gp1]
                            self.up_pedigree_dict[new_parent_id] += [gp2] # move gp2 from parent_id to new_parent_id
                            self.get_down_pedigree_dict_from_up_pedigree_dict()
                            self.update_all_rels()
                            self.compute_pedigree_log_likelihood()
                            self.adjust_sides(focal_id=new_parent_id, ibd_segs=ibd_segs, overlap_length_delta=overlap_length_delta, flag_only=flag_only)
            self.adjust_sides(focal_id=parent_id, ibd_segs=ibd_segs, overlap_length_delta=overlap_length_delta, flag_only=flag_only)
        elif len(self.up_pedigree_dict[focal_id][2:]) == 2:
            # WARNING: this is adding parents of the wrong sex. Maybe add parents by "add_parent_for_child" rather than getting a new number below.
            p1,p2 = self.up_pedigree_dict[focal_id][2:] # parent1,parent2 of focal_id
            info1 = copy.deepcopy(self.up_pedigree_dict.get(p1, self.down_pedigree_dict[p1][:2]))
            info2 = copy.deepcopy(self.up_pedigree_dict.get(p2, self.down_pedigree_dict[p2][:2]))

            # have to do some stuff if p1 or p2 have < 2 parents each
            num_missing_parents1 = 4 - len(info1)
            num_missing_parents2 = 4 - len(info2)

            if num_missing_parents1 + num_missing_parents2 >= 3: # then either there are no grandparents or only one grandparent, which we can't resolve.
                return

            min_parent_ind = self.min_parent_ind

            for count in range(num_missing_parents1):
                new_ind = min_parent_ind - 1
                info1.append(new_ind)
                min_parent_ind -= 1

            for count in range(num_missing_parents2):
                new_ind = min_parent_ind - 1
                info2.append(new_ind)
                min_parent_ind -= 1

            gp11,gp12 = info1[2:]
            gp21,gp22 = info2[2:]

            child_set1 = set(self.down_pedigree_dict.get(gp11,[])[2:]) & set(self.down_pedigree_dict.get(gp12,[])[2:]) # set of children shared in common between the parents of p1
            child_set2 = set(self.down_pedigree_dict.get(gp21,[])[2:]) & set(self.down_pedigree_dict.get(gp22,[])[2:]) # set of children shared in common between the parents of p2

            other_children1 = child_set1 - {p1}
            other_children2 = child_set2 - {p2}
            # if neither p1 nor p2 is genotyped and there are no descendant lineages available for triangulation, look for overlaps and try to rearrange ancestral lineages if necessary
            if (not self.is_genotyped(p1)) and (not self.is_genotyped(p2)) and (not other_children1) and (not other_children2):
                rels_dict = dict()
                rels_dict[gp11] = {rel_id for rel_id in self.rel_dict.get(gp11,{'rel' : set()})['rel'] if self.is_genotyped(rel_id)} # genotyped relatives (excluding descendants and direct ancestors) of gp11
                rels_dict[gp12] = {rel_id for rel_id in self.rel_dict.get(gp12,{'rel' : set()})['rel'] if self.is_genotyped(rel_id)}
                rels_dict[gp21] = {rel_id for rel_id in self.rel_dict.get(gp21,{'rel' : set()})['rel'] if self.is_genotyped(rel_id)}
                rels_dict[gp22] = {rel_id for rel_id in self.rel_dict.get(gp22,{'rel' : set()})['rel'] if self.is_genotyped(rel_id)}

                if self.is_genotyped(gp11):
                    rels_dict[gp11].add(gp11)
                if self.is_genotyped(gp12):
                    rels_dict[gp12].add(gp12)
                if self.is_genotyped(gp21):
                    rels_dict[gp21].add(gp21)
                if self.is_genotyped(gp22):
                    rels_dict[gp22].add(gp22)

                p1_desc_set = set(self.get_descendant_dict(p1).keys()) - {p1} # self.get_descendant_dict includes the target id, so we need to remove it.
                p2_desc_set = set(self.get_descendant_dict(p2).keys()) - {p2}

                p1_desc_set = {desc_id for desc_id in p1_desc_set if self.is_genotyped(desc_id)}
                p2_desc_set = {desc_id for desc_id in p2_desc_set if self.is_genotyped(desc_id)}
                if p1_desc_set == p2_desc_set: # if p1_desc_set != p2_desc_set then there are half siblings/descendants we can triangulate on. So don't look for overlapping segments.
                    p1p2_desc_set = {uid for uid in p1_desc_set if self.is_genotyped(uid)}
                    indep_desc_set = set(self.get_independent_inds(p1p2_desc_set))
                    segs_dict = dict()
                    segs_dict[gp11] = [seg for seg in ibd_segs if (seg[0] in indep_desc_set and seg[1] in rels_dict[gp11]) or (seg[0] in rels_dict[gp11] and seg[1] == indep_desc_set)] # segments shared between gp11 and their relatives (excluding descendants and direct ancestors)
                    segs_dict[gp12] = [seg for seg in ibd_segs if (seg[0] in indep_desc_set and seg[1] in rels_dict[gp12]) or (seg[0] in rels_dict[gp12] and seg[1] in indep_desc_set)]
                    segs_dict[gp21] = [seg for seg in ibd_segs if (seg[0] in indep_desc_set and seg[1] in rels_dict[gp21]) or (seg[0] in rels_dict[gp21] and seg[1] in indep_desc_set)]
                    segs_dict[gp22] = [seg for seg in ibd_segs if (seg[0] in indep_desc_set and seg[1] in rels_dict[gp22]) or (seg[0] in rels_dict[gp22] and seg[1] in indep_desc_set)]

                    # find out if any segments in the rels of gp11 substantially overlap any segments in the rels of gp12 (and compare gp11 vs gp21, gp11 vs gp22, gp21 vs gp12, etc.)
                    overlapping_lengths_dict = dict()
                    gp_set = {gp11,gp12,gp21,gp22}
                    for gpA,gpB in combinations(gp_set, r=2):
                        key = frozenset({gpA,gpB})
                        if key not in overlapping_lengths_dict:
                            overlapping_lengths_dict[key] = 0
                        for segA in segs_dict[gpA]:
                            for segB in segs_dict[gpB]:
                                if segA[2] != segB[2]: # check whether the two segments are on the same chromosome. If not, continue
                                    continue
                                if ((segB[4] >= segA[3]) and (segB[3] <= segA[4])) or ((segA[4] >= segB[3]) and (segA[3] <= segB[4])): # if the two segments overlap
                                    length = min(segA[4],segB[4]) - max(segA[3],segB[3])
                                    overlapping_lengths_dict[key] += length

                    groupings = [(frozenset({gp11,gp12}),frozenset({gp21,gp22})),
                                 (frozenset({gp11,gp21}),frozenset({gp12,gp22})),
                                 (frozenset({gp11,gp22}),frozenset({gp12,gp21}))]

                    current_overlap = 0
                    min_overlap = float('inf')
                    best_grouping = None
                    for pair_key,other_pair_key in groupings:
                        total_within = overlapping_lengths_dict[pair_key] + overlapping_lengths_dict[other_pair_key]
                        if (pair_key,other_pair_key) == (frozenset({gp11,gp12}),frozenset({gp21,gp22})):
                            current_overlap = total_within
                        if total_within < min_overlap:
                            min_overlap = total_within
                            best_grouping = (pair_key,other_pair_key)
                    # if there is no best grouping (i.e., if current_overlap - min_overlap < overlap_length_delta), then don't rearrange, but mark this individual as having indistinguishable grandparental lineages.
                    if current_overlap - min_overlap >= overlap_length_delta: # then the current grouping may be wrong and there is a better one.
                        if flag_only:
                            self.mispaired_parents_set.add(p1)
                            self.mispaired_parents_set.add(p2)
                        else:
                            true_gp_set = set(self.up_pedigree_dict[p1][2:] + self.up_pedigree_dict[p2][2:]) # true gps
                            grp1,grp2 = best_grouping
                            bgp_list1 = list(grp1 & true_gp_set)
                            bgp_list2 = list(grp2 & true_gp_set)
                            if (p1 in self.up_pedigree_dict) or (len(bgp_list1) > 0):
                                self.up_pedigree_dict[p1] = self.up_pedigree_dict[p1][:2] + bgp_list1
                            if (p2 in self.up_pedigree_dict) or (len(bgp_list2) > 0):
                                self.up_pedigree_dict[p2] = self.up_pedigree_dict[p2][:2] + bgp_list2
                            self.get_down_pedigree_dict_from_up_pedigree_dict()
                            self.update_all_rels()
                            self.compute_pedigree_log_likelihood()
                    else:
                        self.ids_with_unresolvable_grandparents.add(focal_id)

            # recurse on parents
            self.adjust_sides(focal_id=p1, ibd_segs=ibd_segs, overlap_length_delta=overlap_length_delta, flag_only=flag_only)
            self.adjust_sides(focal_id=p2, ibd_segs=ibd_segs, overlap_length_delta=overlap_length_delta, flag_only=flag_only)

    def make_plottable(self):
        if len(self.up_pedigree_dict) > 1:
            self.strip_founders()
        self.fill_in_parents()
        self.strip_founders()
        self.assign_sexes()
        self.order_sexes()


def extend_up(
    node_id : int,
    deg : int,
    po : Any,
) -> Tuple[Any, Any, Any]:
    """
    Extend a line of deg individuals up from node_id. 
    Return the resulting pedigree object and the final
    and penultimate ids.

    Args:
        node_id: id of person to extend down from.
        deg: number of individuals in chain to extend up.
        po: pedigree object of the pedigree in which node_id lives.
    """
    prev_id = None
    current_id = node_id
    for d in range(deg):
        pid = po.add_parent_for_child(child_id = current_id)
        prev_id = current_id
        current_id = pid
    return (po, current_id, prev_id)


def extend_down(
    node_id : int,
    deg : int,
    po : Any,
    partner_id : int = None,
) -> Any:
    """
    Extend a line of deg individuals down off of node_id.
    If partner_id is not none, connect the child of node_id
    to partner_id.
    Args:
        node_id: id of person to extend down from.
        deg: number of individuals in chain to extend down.
        po: pedigree object of the pedigree in which node_id lives.
        partner_id: id (if any) of the partner of node_id who has the same offspring 
                    as node_id and no genotyped relatives or ancestors.
    """
    prev_id = None
    current_id = node_id
    for d in range(deg):
        cid = po.add_child_for_parent(parent_id = current_id)
        if current_id == node_id and (partner_id is not None):
            po.connect_parent_child(child_id = cid, parent_id = partner_id)
        prev_id = current_id
        current_id = cid
    return (po, current_id, prev_id)


def merge_pedigrees_on_founder(
    id1 : int,
    id2 : int,
    po1 : Any,
    po2 : Any,
    partner_id1 : int = None,
    partner_id2 : int = None,
    ibd_stat_dict : Dict[FrozenSet[int],Dict[str,Any]] = None,
    pw_log_likes : Dict[int, Dict[int, Dict[Tuple[Any,Any,Any], float]]] = None,
) -> Any:
    """
    Replace id2 in po2 with id1.
    Args:
        id1: id in po1 to merge with id2 in po2.
        id2: id in po2 to merge with id1 in po1.
        po1: pedigree object of the first pedigree.
        po2: pedigree object of the second pedigree.
        partner_id1: id (if any) of the partner of anc_id1 who has the same offspring
                     as anc_id1 and no genotyped relatives or ancestors.
        partner_id2: id (if any) of the partner of anc_id2 who has the same offspring
                     as anc_id2 and no genotyped relatives or ancestors.
        ibd_stat_dict : Dict[FrozenSet[int],Dict[str,Any]] = None,
        pw_log_likes : Dict[int, Dict[int, Dict[Tuple[Any,Any,Any], float]]] = None,
    """
    if not po2.is_founder(id2):
        raise Exception("{} is not a founder.".format(id2))
    if partner_id2 and not po2.is_founder(partner_id2):
        raise Exception("{} is not a founder.".format(partner_id2))

    min_id = min(po1.min_parent_ind,po2.min_parent_ind)
    id_update_dict = po2.update_ungenotyped_inds(min_id-1)

    if id2 in id_update_dict:
        id2 = id_update_dict[id2]
    if partner_id2 and partner_id2 in id_update_dict:
        partner_id2 = id_update_dict[partner_id2]

    id1_sex,id1_age = po1.up_pedigree_dict.get(id1,[None,None])[:2]
    id2_sex,id2_age = po2.up_pedigree_dict.get(id2,[None,None])[:2]
    if id1_sex and id2_sex and id1_sex != id2_sex:
        raise Exception("Sexes disagree for {} and {}.".format(id1,id2))
    if id1_age and id2_age and id1_age != id2_age:
        raise Exception("Ages disagree for {} and {}.".format(id1,id2))
    id1_sex = id2_sex if id2_sex is not None else None
    id1_age = id2_age if id2_age is not None else None
    id2_sex = id1_sex if id1_sex is not None else None
    id2_age = id1_age if id1_age is not None else None
    if id2 > 0:
        po1.replace_individual(old_id=id1, new_id=id2, new_sex=id1_sex, new_age=id1_age)
        id1 = id2
    else:
        po2.replace_individual(old_id=id2, new_id=id1, new_sex=id1_sex, new_age=id1_age)

    if id1 in po2.up_pedigree_dict:
        po2.up_pedigree_dict.pop(id1)

    if partner_id1 and partner_id2:
        partner_id1_sex,partner_id1_age = po1.up_pedigree_dict.get(id1,[None,None])[:2]
        partner_id2_sex,partner_id2_age = po2.up_pedigree_dict.get(id2,[None,None])[:2]
        if partner_id1_sex and partner_id2_sex and partner_id1_sex != partner_id2_sex:
            raise Exception("Sexes disagree for {} and {}.".format(id1,id2))
        if partner_id1_age and partner_id2_age and partner_id1_age != partner_id2_age:
            raise Exception("Ages disagree for {} and {}.".format(id1,id2))
        partner_id1_sex = partner_id2_sex if partner_id2_sex is not None else None
        partner_id1_age = partner_id2_age if partner_id2_age is not None else None
        partner_id2_sex = partner_id1_sex if partner_id1_sex is not None else None
        partner_id2_age = partner_id1_age if partner_id1_age is not None else None
        if partner_id2 > 0:
            po1.replace_individual(old_id=partner_id1, new_id=partner_id2, new_sex=partner_id1_sex, new_age=partner_id1_age)
            partner_id1 = partner_id2
        else:
            po2.replace_individual(old_id=partner_id2, new_id=partner_id1, new_sex=partner_id1_sex, new_age=partner_id1_age)
        if partner_id1 in po2.up_pedigree_dict:
            po2.up_pedigree_dict.pop(partner_id1)

    new_up_dict = po1.up_pedigree_dict
    new_up_dict.update(po2.up_pedigree_dict)

    new_age_dict = copy.deepcopy(po1.age_dict)
    new_age_dict.update(po2.age_dict)

    new_sex_dict = copy.deepcopy(po1.sex_dict)
    new_sex_dict.update(po2.sex_dict)

    return PedigreeObject(
        up_pedigree_dict = new_up_dict, 
        ibd_stats = ibd_stat_dict, 
        pairwise_log_likelihoods = pw_log_likes, 
        sex_dict = new_sex_dict, 
        age_dict = new_age_dict,
    )


def connect_pedigrees_through_founders(
    anc_id1 : int,
    anc_id2 : int,
    po1 : Any,
    po2 : Any,
    deg1 : int,
    deg2 : int,
    partner_id1 : int = None,
    partner_id2 : int = None,
    ibd_stat_dict : Dict[FrozenSet[int],Dict[str,Any]] = None,
    pw_log_likes : Dict[int, Dict[int, Dict[Tuple[Any,Any,Any], float]]] = None,
    num_common_ancs : int = 2,
) -> Any:
    """
    Connect two pedigrees through founders anc_id1 and anc_id2.
    Args:
        anc_id1: id of the founder in pedigree po1 to connect to.
        anc_id2: id of the founder in pedigree po2 to connect to.
        po1: pedigree object of the first pedigree.
        po2: pedigree object of the second pedigree.
        deg1: degree (up) from anc_id1 to the common ancestor with po2.
        deg2: degree (up) from anc_id2 to the common ancestor with po1.
        partner_id1: id (if any) of the partner of anc_id1 who has the same offspring
                     as anc_id1 and no genotyped relatives or ancestors.
        partner_id2: id (if any) of the partner of anc_id2 who has the same offspring
                     as anc_id2 and no genotyped relatives or ancestors.
        ibd_stat_dict: Dict mapping ID1 to ID2 to IBD summary stats between ID1 and ID2.
        pw_log_likes: Dict mapping ID1 to ID1 to a dict mapping relationship tuples to their
                      respective point-predicted likelihoods.
        num_common_ancs: Integer representing the number of common ancestors shared between
                         anc_id1 and anc_id2
    """
    if deg1 == 0 and deg2 == 0:
        new_ped_obj = merge_pedigrees_on_founder(
            anc_id1,
            anc_id2,
            po1,
            po2,
            partner_id1 = partner_id1,
            partner_id2 = partner_id2,
            ibd_stat_dict = ibd_stat_dict,
            pw_log_likes = pw_log_likes
        )
    elif deg1 == 0:
        po1,anc_id1,_ = extend_down(anc_id1, deg2, po1, partner_id1)
        new_ped_obj = merge_pedigrees_on_founder(
            anc_id1,
            anc_id2,
            po1,
            po2,
            partner_id1 = partner_id1,
            partner_id2 = partner_id2,
            ibd_stat_dict = ibd_stat_dict,
            pw_log_likes = pw_log_likes
        )
    elif deg2 == 0:
        po2,anc_id2,_ = extend_down(anc_id2, deg1, po2, partner_id2)
        new_ped_obj = merge_pedigrees_on_founder(anc_id2,
            anc_id1,
            po2,
            po1,
            partner_id1 = partner_id2,
            partner_id2 = partner_id1,
            ibd_stat_dict = ibd_stat_dict,
            pw_log_likes = pw_log_likes
        )
    else:
        po1,anc_id1,prev_id = extend_up(anc_id1, deg1, po1)
        if len(po1.up_pedigree_dict[prev_id][2:]) < 2:
            partner_id = po1.add_parent_for_child(child_id = prev_id)
        else:
            pids = po1.up_pedigree_dict[prev_id][2:]
            partner_id_set = set(pids) - {anc_id1}
            partner_id = partner_id_set.pop()
        if num_common_ancs == 1:
            partner_id = None
        if (partner_id is None) or po1.is_founder(partner_id):
            po1,anc_id1,_ = extend_down(anc_id1, deg2, po1, partner_id)
            new_ped_obj = merge_pedigrees_on_founder(
                anc_id1,
                anc_id2,
                po1,
                po2,
                ibd_stat_dict = ibd_stat_dict,
                pw_log_likes = pw_log_likes
            )
        else:
            new_ped_obj = None
    return new_ped_obj




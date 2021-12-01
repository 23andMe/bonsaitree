bonsai
=============

Algorithm for automatically building pedigrees using IBD, Age, and Sex information.


Installation and testing
------------------------
To install:
```
make
python setup.py install
```

To test:
```
make test
```


Interface
---------
To build a tree for a group of related individuals:
```
from bonsaitree import bonsai

result = bonsai.build_pedigree(ibd_seg_list, bio_info, focal_id=None)
```


Inputs
------
The required inputs to Bonsai are listed below. Additional parameters can be specified by the user. See [Additional Bonsai Parameters](#BonsaiParameters).
* `ibd_seg_list`: (List of lists) A list of IBD segments of the form `[[id1, id2, chrom, start, end, is_full, seg_cm]]`, where the elements are of the following types:
    - `id`,`id2`: (int) IDs of the individuals sharing the segment. Note: Bonsai currently allows only positive integers for genotyped ids. The ungenotyped nodes in a pedigree will be negative integers.
    - `chrom`: (string) Chromosome on which the segment is found.
    - `start`: (float) Physical start position of the segment.
    - `end`: (float) Physical end position of the segment.
    - `is_full`: (bool) Indicates whether the segment is half (IBD1) or full (IBD2).
    - `seg_cm`: (float) Length of the segment in centimorgans.
* `bio_info`: (List of dicts) A list of dicts containing sex and age information for genotyped individuals. `bio_info` has the form [{'genotype_id' : int, 'sex' : 'M'/'F', 'age' : int}].
* `focal_id`: (Optional, int) To force the pedigree builder to start with a specified individual. Pedigrees can be different if they use different starting individuals. This ensures that the individual of interest is placed and it can improve the estimated relationships between them and their close relatives.

An example set of inputs can be found in `tests/fixtures/4gens_2offspring_0.1probhalf.json`. This is a very large pedigree. To see the contents of this file:
```
import json
input_data_path = 'tests/fixtures/4gens_2offspring_0.1probhalf.json'
input_data = json.loads(open(input_data_path).read())
```


Output
------
The output of bonsai is a dictionary containing a variety of information about the pedigree. Keys of the dictionary are as follows:
* `normed_pedigree`: (dict) A dictionary representing the topology of the pedigree presented in a normalized form so that each node has two filled-in parents, the focal id has nodes filled in up to great grandparents, unknown sexes of spouses are imputed if the sex of the other spouse is known, leaf nodes are deleted if they are unrelated to the focal individual, and sexes of parents are ordered with the mother listed first (if sex is known). `normed_pedigree` has the form `normed_pedigree = {child_id : [sex, parent1, parent2]}`.
* `ped_obj`: (instance of the PedigreeObject class) `ped_obj` has attributes that include the inferred pedigree topology, inferred pairwise relationships, the pedigree log likelihood, etc. It contains methods for modifying a pedigree, adding or removing individuals, getting ancestors or descendants of a node, finding common ancestors of a set of nodes, etc. Use `dir(ped_obj)` for a full list of attributes and `ped_obj.attribute?` to see information about `attribute`. Some of the most common attributes are below:

    - `up_pedigree_dict`: (dict) Stores the topology of the inferred pedigree. Has the form `{child_id : [child_sex, child_age, parent1_id, parent2_id]}`.
    - `down_pedigree_dict`: (dict) Stores the topology of the inferred pedigree. Has the form `{parent_id : [parent_sex, parent_age, child1_id, child2_id, ...]}`.
    - `all_ids`: (list) List of all ids in the pedigree.
    - `ibd_stats`: (dict) Dict with keys of the form `frozenset({id1, id2})` and values giving summary statistics of the ibd sharing between the pair.
    - `rel_dict`: (dict) Dict of the form `dict[id] = {'anc' : <Set of ancestor ids>, 'desc' : <Set of descendant ids>, 'rel' : <Set of relatives who are neither direct descendants nor ancestors>}`.
    - `rels`: (dict) Nested dict of the form `dict[id1][id2] = deg`, where `deg` is a three-element tuple representing the relationship between `id1` and `id2`. Deg is of the form `deg = (num_up, num_down, num_anc)`, where `num_up` is the number of meioses separating `id1` from its common ancestor(s) with `id2`. `num_down` is the number of meioses separating `id2` from its common ancestor(s) with `id1`. `num_anc` is the number of common ancestors shared between `id1` and `id2`.
    - `pairwise_log_likelihoods`: (dict) Nested dict of the form `dict[id1][id1] = log_like`, where `log_like` is the pairwise log likelihood of the relationship between `id` and `id2` based on IBD sharing and age.

    - `get_connecting_path_set`: (method) Find all ancestors on the path connecting two related nodes (`id1` and `id2`). 
        - Usage: `path_set = ped_obj.get_connecting_path_set(id1, id2)`.
    - `is_founder`: (method) Return a bool describing whether individual `id` is a pedigree founder.
        - Usage: `ped_obj.is_founder(id)`.
    - `is_leaf`: (method) Return a bool describing whether individual `id` is a leaf node.
         - Usage: `ped_obj.is_leaf(id)`.
    - `keep_nodes`: (method) For a set of node ids `S`, remove all nodes that are not in `S`, or which don't lie on a path connecting some pair in this set. The optional boolean parameter `include_parents` retains all parents of nodes in `S`, even if they don't lie on a path connecting two nodes in `S`.
        - Usage: `ped_obj.keep_nodes({id1, id2, ...}, include_parents=False)`.
    - `order_sexes`: (method) Order sexes of parent ids in `ped_obj.up_dict` so that the maternal id (if known) appears first and the paternal id appears second.
        - Usage: `ped_obj.order_sexes()`.

WARNING: 
* If there are twins or duplicated individuals (with different ids) in the input data, the pedigree in `ped_obj` only places one such individual and sets the others to the side. These unplaced twins are added back in the `normed_pedigree` dictionary, or they can be accessed using `result['twin_dict']`, which is a dictionary mapping the placed twin id to all of its twin ids.



<a name="BonsaiParameters"></a>Additional Bonsai Parameters
------------------
* **seed_pedigree_list**
    -   default = ()
    -   optional ist of seed pedigrees to use as starting points for building the pedigree.
* **validated_node_set_list**
    -   default = ()
    -   optional list of validated nodes. Only specify this if you have specified `seed_pedigree_list`. All other nodes will be removed from pedigrees in seed_pedigree_list before using these pedigrees as sarting points for new pedigrees.
* **ignore_validated**
    -   default = True
    -   set to False to use seed pedigrees and validated nodes. Makes it possible build from scratch when using input data that contains seed pedigrees.
* **disallow_distant_half_rels**
    -   default = True
    -   do not place relatives more distant than first cousin through a single common ancestor.
* **max_radius** 
     -  default = float('inf') 
     -  Bonsai first infers small pedigrees and then assembles them together. `max_radius` is the degree from the focal individual used to seed each small pedigree to their most distant placed relative before combining small pedigrees together.
* **max_add_degree**
    -   default = 4
    -   when building each small pedigree, stop adding individuals when no unplaced individual has a placed relative with degree <= max_add_degree.
* **min_rel_append_types**
    -   default = 1
    -   when buiding each small pedigree, try to place each new individual in all ways consistent with at least the `top min_rel_append_types` most likely point-predicted relationships with their closest relative.
* **max_rel_append_types**
    -   default = 3
    -   when building each small pedigree, try to place each new individual in all ways consistent with at most the top `max_rel_append_types` most likely point-predicted relationships with their closest relative.
* **ibd_threshold**
    -   default = 0
    -   remove all ibd segments shorter than ibd_threshold. Ignored if ibd_threshold=0.
* **remove_distant_threshold**
    -   default = 8
    -   remove all placed genotyped individuals if their degree (a,b,c) to the focal individual satisfies b > `remove_distant_threshold`.
* **ped_save_like_abs_fraction**
    -   default = 0.01
    -   when building small pedigrees one person at a time, Bonsai must decide how many different alternate pedigrees to keep around after each person is added. Retaining fewer alternate pedigrees makes Bonsai faster, but performs a less thorough search of the pedigree space. The `ped_save_like_abs_fraction` parameter is one of two parameters controling how many pedigrees are retained at each step. Let p^1, p^2, ..., p^K denote K alternate pedigrees at a given step, ordered in likelihood from most to least likely. If, for the kth pedigree, L(p^(k)) >= L(p^(1)) * ped_save_like_abs_fraction, we retain it.
* **ped_save_like_delta_fraction**
    -   default = 0.001
    -   when building small pedigrees one person at a time, Bonsai must decide how many different alternate pedigrees to keep around after each person is added. Retaining fewer alternate pedigrees makes Bonsai faster, but performs a less thorough search of the pedigree space. The `ped_save_like_delta_fraction` parameter is one of two parameters controling how many pedigrees are retained at each step. Let p^1, p^2, ..., p^K denote K alternate pedigrees at a given step, ordered in likelihood from most to least likely. If, for the kth pedigree, L(p^(k)) >= L(p^(k-1)) * ped_save_like_delta_fraction, we retain it.
* **num_small_ped_objs_to_save**
    -   default = 10
    -   when building small pedigrees, retain at most `num_small_ped_objs_to_save` pedigrees, subject to the constraints of `ped_save_like_abs_fraction` and `ped_save_like_delta_fraction`.
* **drop_ibd_alpha**
    -   default = 1e-4
    -   p-value threshold for hypothesis test for dropping background IBD.

IBD inference
---------
IBD can be inferred using any method. However, Bonsai was developed using an in-house method for IBD inference that estimates IBD from unphased data ([Henn et al., 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0034267)). A publicly available method that behaves very similarly is Ibis ([Seidman et al., 2020](https://github.com/williamslab/ibis)). The pre-trained distributions that come packaged with Bonsai have been tested with Ibis.  Code for converting Ibis output to Bonsai input is included with Bonsai and an example of usage is shown below.


Example usage
---------
The following is an example of how to read segment data from Ibis and build a Bonsai tree. Running Ibis will generate an output directory of `.seg` files, each of which contains IBD segments inferred for a group of individuals for a given chromosome (e.g., `1.seg`, `2.seg`, ..., `22.seg`). Let's suppose the directory is called `ibis_output_dir`. Bonsai contains a script for reading Ibis output and converting it to a list of segments that can be used as input for Bonsai. To load the Ibis segments in Python:
```
from bonsaitree.utils import read_ibis_ibd

ibd_seg_list = read_ibis_ibd('ibis_output_dir')
```
This creates a list `ibd_seg_list` of tuples, each of which is an IBD segment. We can inspect the first three elements of this list as follows:
```
ibd_seg_list[:3]
```
which will produce something like
```
[
(1, 2, '22',  1_502_003, 50_152_011, False, 72.3)
(1, 7, '10',    414_217,  1_222_982, False, 30.9)
(1, 2, '22', 10_928_382, 49_100_112, True,  50.1)
]
```
Each tuple represents an IBD segment of the form
```
(id1, id2, 'chrom', phys_start, phys_end, is_full, len_in_cm)
```
where the ids are the ones you chose when computing the IBD in Ibis.

In addition to the `ibd_seg_list`, Bonsai has one other required input, `bio_info`, and an optional input `focal_id` (along with other optional parameters discussed above). The `bio_info` object is a list of dicts of the following form:
```
bio_info = [
    {'genotype_id' : id, 'age' : age, 'sex' : sex},
    ...
]
```
and it should contain one entry for each individual in the pedigree. The `focal_id` allows you to specify a particular individual for whom you wish to build the pedigree. If `focal_id` is specified, the first individual placed will be `focal_id` and the resulting pedigree is guaranteed to contain this individual. If `focal_id` is left unspecified, Bonsai will choose a focal person to start with. This person is the individual who shares the most IBD on average with all others.

Bonsai can then be run using
```
from bonsaitree import bonsai

result = bonsai.build_pedigree(ibd_seg_list, bio_info, focal_id=focal_id)
```
The result is a dictionary containing many objects. Perhaps the most useful is `ped_obj`. This is an instance of the `PedigreeObject` class and it contains the inferred pedigree topology, along with functions that allow you to perform various computations on the inferred pedigree. If you just want the pedigree topology, it can be found as follows:
```
ped_obj = result['ped_obj']
ped_obj.up_pedigree_dict
```
The `up_pedigree_dict` is a dictionary mapping each node id in the pedigree to its sex, age, parent1 and parent2 (see the list of outputs above). It has one entry for every node in the pedigree. This object specifies the full topology of the pedigree. Parents are not ordered by sex, as this sex is generally unknown for inferred nodes. Additional operations that can be computed using the pedigree object include 
1. Finding the relationship between a pair of nodes (say, `id1` and `id2`):

        ped_obj.rels[id1][id2]

2. Finding all individuals on the path from `id1` to `id2`:

       ped_obj.get_connecting_path_set(id1,id2)

3. Checking if an individual is a leaf node or a founder node

        ped_obj.is_leaf(id)
        ped_obj.is_founder(id)

Many more operations are available. Type `dir(ped_obj)` for a list of attributes and methods.

Updating age and IBD-sharing distributions
---------
Bonsai uses empirically determined means and standard deviations for IBD sharing and age differences that are used for predicting pairwise relationships. The means and standard deviations can be found in the `bonsaitree.models` directory. IBD sharing moments can be found in `distn_dict.json`, which contains a json serialized dictionary of the following form

```
{
    rel_tuple : {
        summary_stat : {
            1 : [IBD1_mean, IBD1_std_dev],
            2 : [IBD2_mean, IBD2_std_dev],
        }
        ...
    }
    ...
}
```

Here, `rel_tuple` is a tuple representing a relationship between a pair of individuals (`i` and `j`). `rel_tuple` is of the form `(up, down, num_ancs)`, where `up` is the number of meioses between individual `i` and their most recent common ancestor(s) with `j`, `down` is the number of meioses between `j` and their most recent common ancestor(s) with `i` and `num_ancs` is the number of most recent common ancestors between `i` and `j` (either 1 or 2). To reduce the computational burden of fitting distributions, Bonsai also accepts tuples of the form `(up+down, num_ancs)`, or a mixture of the two formats. The `summary_stat` is either `count` or `total_len`. The `summary_stat` is either `count` or `total_len`, and stores the mean and standard deviation of the statistic for both IBD1 (`1`) and IBD2 (`2`). 

Means and standard deviations for pairwise age differences can be found in `age_diff_moments.json`, which contains a json serialized dictionary of the following form:

```
{
    rel_tuple : [age_diff_mean, age_diff_std_dev]
    ...
}
```
where `rel_tuple` is the tuple denoting the relationship between a pair of indiviudals `i` and `j`, which is of the form described previously. The quantities `age_diff_mean` and `age_diff_std_dev` are the mean and standard deviation of the difference `age(j) - age(i)`.
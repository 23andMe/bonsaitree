ttam.bonsai.v3
=============

Version 3 of the Bonsai algorithm for automatically building pedigrees
using IBD, age, and sex information.


Overview
---------
Bonsai v3 provides methods to
1. Build a pedigree from scratch.
2. Add a new individual to an existing pedigree and obtain a list of possible pedigrees that include the new individual and their likelihoods.
3. Add a new individual to an existing pedigree and obtain a list of possible ways of connecting the new individual and their likelihoods.


<!-- Note
----
Bonsai v3 can run with either phased IBD segments or unphased IBD segments. To get the best of both worlds, you can use a combination of both:
phased IBD segments for distant relatives and unphased IBD segments for close relatives. If you input both phased and unphased IBD to Bonsai,
it will used the unphased IBD segments among all pairs for whom they are available and it will use the phased IBD segments for all other pairs.

For example. Suppose that you have computed unphased IBD among individuals `A` and `B` and phased IBD among all three individuals `A`, `B` and `C`.
If you provide these two segment sets to Bonsai, it will use the unphased IBD among `A` and `B` and it will use the phased IBD between pairs (`A`, `C`)
and (`B`,`C`). -->


Usage
------
1. `build_pedigree()`: Build a pedigree from scratch

    ```python
    from ttam.bonsai.v3.bonsai import build_pedigree

    up_dict_log_like_list = build_pedigree(
        bio_info=bio_info,
        unphased_ibd_seg_list=unphased_ibd_seg_list,
    )
    ```
    **Input**
    * `bio_info`: A list of dicts of the form `[{'genotype_id': <int>, 'age': <int>, 'sex': <'M' or 'F'>, 'coverage': float},...]`. For full coverage, the `coverage` value should be set to `float('inf')`.
    * `unphased_ibd_seg_list`: A list of the form `[[id1, id2, chrom, start_bp, end_bp, is_full, len_cm],...]` where
      * `id1`: The ID (`int`) of the first person.
      * `id2`: The ID (`int`) of the second person.
      * `chrom`: The chromosome (`str` or `int`) on which the segment is found.
      * `start_bp`: Physical start position (`float`) of the segment.
      * `end_bp`: Physical end position (`float`) of the segment.
      * `is_full`: Boolean indicating whether the segment is `full` (i.e., IBD2). If `is_full` is `False`, the segment is a "half or full" ( NOT IBD1 ).
      * `len_cm`: Length of the segment in cM (`float`).
    * [Optional] `phased_ibd_seg_list`: A list of the form `[[id1, id2, hap1, hap2, chrom, start_cm, end_cm, len_cm],...]` where
      * `id1`: The ID (`int`) of the first person.
      * `id2`: The ID (`int`) of the second person.
      * `hap1`: Haplotype in `id1` on which the segment occurs (`0` or `1`).
      * `hap2`: Haplotype in `id2` on which the segment occurs (`0` or `1`).
      * `chrom`: The chromosome (`int`) on which the segment is found.
      * `start_cm`: Genetic start position (`float`) of the segment in cM.
      * `end_cm`: Genetic end position (`float`) of the segment in cM.
      * `len_cm`: Length of the segment in cM (`float`).
    **Output**
    * A list of tuples of the form `[(up_node_dict, log_like),...]`:
      * `up_node_dict`: a dictionary that stores each inferred pedigree structure of the form `up_node_dict = {node : {parent1 : degree1, <parent2> : <degree2>}, ...}`.
      * `log_like`: the (composite) log likelihood of the pedigree structure.


2. `connect_new_node_many_ways()`: Add a new node to an existing pedigree and return multiple pedigrees, each of which contains a different way of placing the node.

    ```python
    from ttam.bonsai.v3.bonsai import connect_new_node_many_ways

    up_dict_log_like_list = connect_new_node_many_ways(
        node=node,
        up_node_dict=up_node_dict,
        bio_info=bio_info,
        unphased_ibd_seg_list=unphased_ibd_seg_list,
        max_peds=max_peds,
    )
    ```
    **Input**
    * `node`: ID of the new node to connect (`int`).
    * `up_node_dict`: Dictionary specifying the pedigree topology by specifying the parent nodes of each node. `up_node_dict` has the form `up_node_dict = {node : {parent1 : degree1, <parent2> : <degree2>}, ...}`.
    * `bio_info`: A list of dicts of the form `[{'genotype_id': <int>, 'age': <int>, 'sex': <'M' or 'F'>, 'coverage': float},...]`. For full-coverage individuals, the `coverage` value should be set to `float('inf')`.
    * `unphased_ibd_seg_list`: A list of the form `[[id1, id2, chrom, start_bp, end_bp, is_full, len_cm],...]` where
      * `id1`: The ID (`int`) of the first person.
      * `id2`: The ID (`int`) of the second person.
      * `chrom`: The chromosome (`str` or `int`) on which the segment is found.
      * `start_bp`: Physical start position (`float`) of the segment.
      * `end_bp`: Physical end position (`float`) of the segment.
      * `is_full`: Boolean indicating whether the segment is `full` (i.e., IBD2). If `is_full` is `False`, the segment is a "half or full" segment as specified by IBD64.
      * `len_cm`: Length of the segment in cM (`float`).
    * [Optional] `phased_ibd_seg_list`: A list of the form `[[id1, id2, hap1, hap2, chrom, start_cm, end_cm, len_cm],...]` where
      * `id1`: The ID (`int`) of the first person.
      * `id2`: The ID (`int`) of the second person.
      * `hap1`: Haplotype in `id1` on which the segment occurs (`0` or `1`).
      * `hap2`: Haplotype in `id2` on which the segment occurs (`0` or `1`).
      * `chrom`: The chromosome (`int`) on which the segment is found.
      * `start_cm`: Genetic start position (`float`) of the segment in cM.
      * `end_cm`: Genetic end position (`float`) of the segment in cM.
      * `len_cm`: Length of the segment in cM (`float`).
    * `max_peds`: Maximum number of pedigrees to return (i.e., maximum number of ways to connect the node).
    **Output**
    * A list of tuples of the form `(up_node_dict, log_like)`, each representing a different way of placing the new node:
      * `up_node_dict`: a dictionary that stores each inferred pedigree structure of the form `up_node_dict = {node : {parent1 : degree1, <parent2> : <degree2>}, ...}`.
      * `log_like`: the (composite) log likelihood of the pedigree structure.


3. `get_new_node_connections()`: Find all ways of placing a new node `node` into an existing pedigree, but do not actually place the node.

    ```python
    from ttam.bonsai.v3.bonsai import get_new_node_connections

    point_tuple_log_like_list = get_new_node_connections(
        node=node,
        up_node_dict=up_node_dict,
        bio_info=bio_info,
        unphased_ibd_seg_list=unphased_ibd_seg_list,
    )
    ```

    **Input**
    * `node`: ID of the new node to connect (`int`).
    * `up_node_dict`: Dictionary specifying the pedigree topology by specifying the parent nodes of each node. `up_node_dict` has the form `up_node_dict = {node : {parent1 : degree1, <parent2> : <degree2>}, ...}`.
    * `bio_info`: A list of dicts of the form `[{'genotype_id': <int>, 'age': <int>, 'sex': <'M' or 'F'>, 'coverage': float},...]`. For full-coverage individuals, the `coverage` value should be set to `float('inf')`.
    * `unphased_ibd_seg_list`: A list of the form `[[id1, id2, chrom, start_bp, end_bp, is_full, len_cm],...]` where
      * `id1`: The ID (`int`) of the first person.
      * `id2`: The ID (`int`) of the second person.
      * `chrom`: The chromosome (`str` or `int`) on which the segment is found.
      * `start_bp`: Physical start position (`float`) of the segment.
      * `end_bp`: Physical end position (`float`) of the segment.
      * `is_full`: Boolean indicating whether the segment is `full` (i.e., IBD2). If `is_full` is `False`, the segment is a "half or full" segment as specified by IBD64.
      * `len_cm`: Length of the segment in cM (`float`).
    * [Optional] `phased_ibd_seg_list`: A list of the form `[[id1, id2, hap1, hap2, chrom, start_cm, end_cm, len_cm],...]` where
      * `id1`: The ID (`int`) of the first person.
      * `id2`: The ID (`int`) of the second person.
      * `hap1`: Haplotype in `id1` on which the segment occurs (`0` or `1`).
      * `hap2`: Haplotype in `id2` on which the segment occurs (`0` or `1`).
      * `chrom`: The chromosome (`int`) on which the segment is found.
      * `start_cm`: Genetic start position (`float`) of the segment in cM.
      * `end_cm`: Genetic end position (`float`) of the segment in cM.
      * `len_cm`: Length of the segment in cM (`float`).
    * `max_peds`: Maximum number of pedigrees to return (i.e., maximum number of ways to connect the node).

    **Output**
    * `point_tuple_log_like_list`: A list of the form `[[con_pt, rel_tuple, log_like],...]` with the following elements:
      * `con_pt`: The point in `up_node_dict` through which the new node `node` is connected. `con_pt` is a tuple of the form `(id, partner, direction)`, where
        * `id`: The ID in `up_node_dict` to which `node` is connected,
        * `partner`: An optional partner of `id` to which `node` is also connected (`partner` = `None` if `node` is not connected to a partner of `id`) and
        * `direction`: An integer indcating the direction of the connection from `id` to `node`. `direction` is
          * `1` if the connection from `id` to `node` extends upward from `id`,
          * `0` if the connection extends downard from `id` and
          * `None` if `node` fits into the position occupied by `id`.
      * `rel_tuple`: A tuple of the form (`deg1`, `deg1`, `num_ancs`), where
        * `deg1`: The degree up from `id` in `con_pt` to its common ancestor with `node`.
        * `deg2`: The degree up from `node` in `con_pt` to its common ancestor with `id`.
        * `num_ancs`: The number of common ancestors shared by `id` and `node`.
      * `log_like1`: Log likelihood (composite) of the way of placing `node`.


4. `connect_new_node()`: Connect a new node `new_node` to an existing pedigree and return the new pedigree with the connected node.

    ```python
    from ttam.bonsai.v3.bonsai import connect_new_node

    new_up_node_dict = connect_new_node(
        new_node=new_node,
        up_node_dict=up_node_dict,
        con_pt=con_pt,
        rel_tuple=rel_tuple,
    )
    ```
    **Input**
    * `node`: ID of the new node to connect (`int`).
    * `up_node_dict`: Dictionary specifying the pedigree topology by specifying the parent nodes of each node. `up_node_dict` has the form `up_node_dict = {node : {parent1 : degree1, <parent2> : <degree2>}, ...}`.
    * `con_pt`: Point of the form `con_pt = (id, partner_id, direction)` in `up_node_dict` through which the pedigree is connected to `node`. `con_pt` is returned as the first element in each tuple from `get_new_node_connections()`.
    * `rel_tuple`: The relationshp between `con_pt` and `node`. Returned as the second element in each tuple from `get_new_node_connections()`.

    **Output**
    * `new_up_node_dict`: An up node dict of the form `{node: {parent_id1: <int>, <parent_id2>: <int>},...}` in which `new_node` has been placed
      into `up_node_dict`.


5. `get_next_node()`: Find the next node to add to the pedigree. Finds the node that shares the most IBD with someone already placed.

    ```python
    from ttam.bonsai.v3.bonsai import get_next_node

    next_node = get_next_node(
        placed_id_set=placed_id_set,
        unphased_ibd_seg_list=unphased_ibd_seg_list,
    )
    ```

    **Input**
    * `placed_id_set`: Set of genotyped IDs already placed in a pedigree.
    * `unphased_ibd_seg_list`: A list of the form `[[id1, id2, chrom, start_bp, end_bp, is_full, len_cm],...]` where
      * `id1`: The ID (`int`) of the first person.
      * `id2`: The ID (`int`) of the second person.
      * `chrom`: The chromosome (`str` or `int`) on which the segment is found.
      * `start_bp`: Physical start position (`float`) of the segment.
      * `end_bp`: Physical end position (`float`) of the segment.
      * `is_full`: Boolean indicating whether the segment is `full` (i.e., IBD2). If `is_full` is `False`, the segment is a "half or full" segment as specified by IBD64.
      * `len_cm`: Length of the segment in cM (`float`).

    **Output**
    * `next_node`: ID (`int`) of the next node to place. This is the node that shares the most IBD with anyone in `placed_id_set`.



Bonsai Parameters
------------------
* **min_seg_len**
    - _Definition_: The minimum observable segment length (e.g., 7cM)
    - _Type_: `float`
    - _Default_: `ttam.bonsai.v3.constants.MIN_SEG_LEN`
* **max_con_pts**
    - _Definition_: The maximum number of points in each pedigree to explore when determining the points through which it can be connected to another pedigree or to a new node.
    - _Type_: `int`
    - _Default_: `ttam.bonsai.v3.constants.MAX_CON_PTS`
* **restrict_connection_points**
    - _Definition_: When combining two pedigrees or when combining a pedigree with a new node, consider only connection points in a pedigree that are within the subtree spanning the genotyped nodes that share IBD with the other pedigree (or node).
    - _Type_: `bool`
    - _Default_:  `ttam.bonsai.v3.constants.RESTRICT_CON_PTS`
* **connect_up_only**
    - _Definition_: When connecting two pedigrees, only connect them "upward" through a common ancestor in each pedigree who is the common ancestor of all genotyped nodes that share IBD with the other pedigree. If no such set exists, attempt to find the best set.
    - _Type_: `bool`
    - _Default_: `ttam.bonsai.v3.constants.CONNECT_UP_ONLY`
* **max_peds**
    - _Definition_: When combining two pedigrees, or when combining a pedigree and a new node, retain the most likely `max_peds` pedigrees.
    - _Type_: `int`
    - _Default_: `ttam.bonsai.v3.constants.MAX_PEDS`

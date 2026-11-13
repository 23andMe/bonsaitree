bonsai
=============

Algorithm for automatically building pedigrees using IBD, age, and sex information.

This repository contains Bonsai v3, the current version of Bonsai, in `bonsaitree.v3`. New users should use Bonsai v3, as described below.

> **Legacy Bonsai v2:** The top-level `bonsaitree` package (`from bonsaitree import bonsai`) is the earlier Bonsai v2. It remains available so that existing code that depends on it continues to work. Its documentation is in [README_v2.md](README_v2.md).


Installation and testing
------------------------
To install: in the bonsaitree directory containing the file Makefile, type
```
make install
```
at a command prompt. See the [Troubleshooting](#Troubleshooting) section below for common errors.

To test: in the directory containing the file Makefile, type
```
make test
```


Quick start
-----------
To build a pedigree for a group of related individuals:
```python
from bonsaitree.v3.bonsai import build_pedigree

up_dict_log_like_list = build_pedigree(
    bio_info=bio_info,
    unphased_ibd_seg_list=unphased_ibd_seg_list,
)

# the most likely pedigree and its log likelihood
up_node_dict, log_like = up_dict_log_like_list[0]
```


Inputs
------
* `bio_info`: (List of dicts) One dict per genotyped individual, of the form
  ```
  [{'genotype_id': <int>, 'age': <int or float>, 'sex': <'M' or 'F'>, 'coverage': <float>}, ...]
  ```
    - `genotype_id`: (int) ID of the individual. Genotyped IDs must be positive integers. Ungenotyped individuals that Bonsai adds to a pedigree are given negative IDs.
    - `age`: (int or float) Age in years, where a larger value means an older person. Bonsai uses only the *differences* between ages, so ages can be measured from any common reference point. Do not pass birth years, because they reverse the direction of every age difference. If an age is unknown, set it to `None`.
    - `sex`: (str) `'M'` or `'F'`.
    - `coverage`: (float) Genome coverage of the individual. For full-coverage genotypes (e.g., from a genotyping array), set this to `float('inf')`.
* `unphased_ibd_seg_list`: (List of lists) IBD segments of the form `[[id1, id2, chrom, start_bp, end_bp, is_full, len_cm], ...]`, where
    - `id1`, `id2`: (int) Genotype IDs of the two individuals sharing the segment.
    - `chrom`: (str or int) Chromosome on which the segment is found.
    - `start_bp`, `end_bp`: (float) Physical start and end positions of the segment.
    - `is_full`: (bool) `True` if the segment is IBD2. If `False`, the segment is "half or full" (i.e., it is not necessarily IBD1).
    - `len_cm`: (float) Length of the segment in cM.
* `phased_ibd_seg_list`: (Optional, list of lists) Phased IBD segments of the form `[[id1, id2, hap1, hap2, chrom, start_cm, end_cm, len_cm], ...]`, where `hap1` and `hap2` (`0` or `1`) are the haplotypes in `id1` and `id2` on which the segment occurs. Either `unphased_ibd_seg_list` or `phased_ibd_seg_list` must be provided.

Commonly adjusted parameters include `min_seg_len` (the minimum segment length, in cM, that your IBD detection method reports) and `max_peds` (the number of alternative pedigrees retained at each step). See the [Bonsai v3 README](bonsaitree/v3/README.md#bonsai-parameters) for the full list.


Output
------
`build_pedigree()` returns a list of the form `[(up_node_dict, log_like), ...]`, sorted from most to least likely, where
* `up_node_dict`: (dict) The inferred pedigree, mapping each individual to their parents: `{node: {parent1: 1, parent2: 1}, ...}`. A node may have zero, one, or two parents listed. Positive IDs are the genotyped individuals from `bio_info`; negative IDs are inferred ungenotyped ancestors and relatives.
* `log_like`: (float) The composite log likelihood of the pedigree.


Other functions
---------------
Bonsai v3 can also add a new individual to an existing pedigree. These functions are in `bonsaitree.v3.bonsai`:
* `connect_new_node_many_ways()`: Add a new individual to an existing pedigree and return the most likely resulting pedigrees.
* `get_new_node_connections()`: List the possible ways of connecting a new individual to an existing pedigree, with their likelihoods, without connecting them.
* `connect_new_node()`: Connect a new individual to an existing pedigree in a specified way.
* `get_next_node()`: Find the unplaced individual who shares the most IBD with someone already in the pedigree.

See the [Bonsai v3 README](bonsaitree/v3/README.md) for the inputs and outputs of each function.


<a name="Troubleshooting"></a>Troubleshooting
---------
1. If you experience an error running "make install", try upgrading pip before trying other things. To do this, type "pip install --upgrade pip" at a command prompt.
2. If upgrading pip does not work, you can try installing Bonsai in a virtual environment. To create and activate a virtual environment, type "python3 -m venv venv && source venv/bin/activate" at a command prompt. Then try running "make install" again.

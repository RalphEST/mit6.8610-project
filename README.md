### MIT 6.8610 Course Project
# Team: NLProteins
Ralph Estanboulieh, Yepeng Huang, Shuvom Sadhuka, Shashata Sawmya

## Description
Using Transformers, transfer learning, and biologically-informed attention to predict clinical phenotypes from genomic variation in the UK Biobank.

## Documentation
### Using ESM-2
**Installation:**

Some text here.

**Basic usage:** 

Some text here (refer to Jupyter notebook).

**Functions:**

`esm2_embedding(filepath, submodel, ...)`:
* Inputs:
  - `filepath` (_string_): path to file containing one-hot encoding matrix of sequence data for a single protein. Data must be of dimensions $N\times L\times 20$.
  - `submodel` (_string_): ESM-2 submodel. One of `"esm2_t6_8M_UR50D"`, `"esm2_t12_35M_UR50D"`, `"esm2_t30_150M_UR50D"`, `"esm1_t6_43M_UR50S"`, `"esm1_t12_85M_UR50S"`.
  - `...`



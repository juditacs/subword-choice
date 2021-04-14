Code and result tables for the ["Subword Pooling Makes a Difference"](https://arxiv.org/abs/2102.10864) EACL2021 paper

# Prerequisites

The source code for running the experiments is available in the [probing](https://github.com/juditacs/probing) package.
This repository only contains the configuration files, the results tables and the analysis notebooks.

# Running a single experiment

    python $PROBING_PATH/train.py -c config/morphology.yaml --train data/morphology/gender_noun/German/train.tsv --dev data/morphology/gender_noun/German/dev.tsv

# Running multiple experiments

Run all morphology experiments (6th layer only):

    python $PROBING_PATH/train_many_configs.py -c config/morphology.yaml -p config/generate_morphology.py

Run all POS experiments (6th layer only):

    python $PROBING_PATH/train_many_configs.py -c config/pos.yaml -p config/generate_pos.py


# Cite

```
@inproceedings{Acs:2020,
    author    = {Judit \'Acs and \'Akos K\'ad\'ar and Andr\'as Kornai},
    title     = {Subword Pooling Makes a Difference},
    booktitle = {Proceedings of the 16th Conference of the European Chapter of the
        Association for Computational Linguistics, {EACL} 2021}
    publisher = {Association for Computational Linguistics},
    year      = {2021},
}
```

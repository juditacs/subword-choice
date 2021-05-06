#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import logging
import pandas as pd
import os

from probing.config import Config


def generate_configs(config_fn):
    this_dir = os.path.dirname(__file__)
    data_root = os.path.join(this_dir, "..", "data", "pos")
    models = {
        'bert-base-multilingual-cased': 13,
        'xlm-roberta-base': 13,
    }
    for language in ['French', 'Czech', 'German', 'Korean']:
        train_file = f"{data_root}/{language}/train"
        dev_file = f"{data_root}/{language}/dev"
        config = Config.from_yaml(config_fn)
        config.layer_pooling = 6
        config.model_name = 'xlm-roberta-base'
        config.subword_pooling = 'attn'
        config.train_file = train_file
        config.dev_file = dev_file
        config.train_size = 10000
        config.dev_size = 2000
        yield config
        return
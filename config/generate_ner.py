#! /usr/bin/env python
# -*- coding: utf-8 -*-
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
    data_root = os.path.join(this_dir, "..", "data", "ner")
    models = {
        'bert-base-multilingual-cased': 13,
        'xlm-roberta-base': 13,
    }
    for model, layers in models.items():
        for lang_path in os.scandir(data_root):
            language = lang_path.name
            train_file = os.path.join(data_root, language, 'train')
            dev_file = os.path.join(data_root, language, 'dev')
            subword_choices = ['lstm', 'first', 'f+l', 'last', 'last2', 'avg', 'sum', 'max', 'attn']
            for probe in subword_choices:
                logging.info("=================================================")
                logging.info(f"=== {language} {model} {probe} ===")
                logging.info("=================================================")
                config = Config.from_yaml(config_fn)
                config.pool_layers = 6
                config.model_name = model
                config.subword_pooling = probe
                config.train_file = train_file
                config.dev_file = dev_file
                if probe == 'lstm':
                    config.batch_size = 32
                yield config
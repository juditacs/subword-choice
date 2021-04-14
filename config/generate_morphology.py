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
    data_root = os.path.join(this_dir, "..", "data", "morphology")
    models = {
        'bert-base-multilingual-cased': 13,
        'xlm-roberta-base': 13,
    }
    for model, layers in models.items():
        for task_path in os.scandir(data_root):
            task = task_path.name 
            for lang_path in os.scandir(task_path.path):
                language = lang_path.name
                train_file = os.path.join(data_root, task, language, 'train.tsv')
                dev_file = os.path.join(data_root, task, language, 'dev.tsv')
                subword_choices = ['first', 'f+l', 'last', 'last2', 'avg', 'sum', 'max', 'attn', 'lstm']
                for probe in subword_choices:
                    logging.info("=================================================")
                    logging.info(f"=== {language} {task} {model} {probe} ===")
                    logging.info("=================================================")
                    config = Config.from_yaml(config_fn)
                    config.pool_layers = 6
                    config.model_name = model
                    config.subword_pooling = probe
                    config.train_file = train_file
                    config.dev_file = dev_file
                    yield config
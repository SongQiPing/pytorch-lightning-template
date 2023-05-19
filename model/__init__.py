# Copyright 2021 Zhongyang Zhang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .model_interface import MInterface

import importlib
import inspect


def build_model_module(config):
    model_name = config['model_name']
    camel_name = ''.join([i.capitalize() for i in model_name.split('_')])
    try:
        model_module = getattr(importlib.import_module(
            'model.' + model_name, package=__package__), camel_name)
    except ImportError:
        # clean-up
        raise ValueError(
            f'Invalid Model File Name or Invalid Class Name model.{model_name}.{camel_name}')
    # Instancialize a model using the corresponding parameters
    class_args = inspect.getfullargspec(model_module.__init__).args[1:]
    inkeys = config.keys()
    model_args = {}
    for arg in class_args:
        if arg in inkeys:
            model_args[arg] = config[arg]
    return model_module(**model_args)


def build_experiment_module(model, config):
    exp_name = config['experiment_name']
    camel_name = ''.join([i.capitalize() for i in exp_name.split('_')])
    try:
        exp_module = getattr(importlib.import_module(
            'model.' + exp_name, package=__package__), camel_name)
    except ImportError:
        # clean-up
        raise ValueError(
            f'Invalid Experiment File Name or Invalid Class Name model.{exp_name}.{camel_name}')
    return exp_module(model, config)

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _tutorial-from-mxnet:

Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_, \
            `Kazutaka Morita <https://github.com/kazum>`_

This article is an introductory tutorial to deploy mxnet models with Relay.

Code Modified for op-profiling by :
Rohan Mukherjee <https://github.com/rohan2606>

"""

import sys

from utils import get_image, get_opt_params
import run_mxnet_on_tvm
import json
from collections import defaultdict

##############################################################
# Get the profiling hyper-parameters like target hardware, ctx and optimization level from input
target, ctx, input_opt_level = get_opt_params(sys.argv)

##############################################################
# Get the test image
x, synset = get_image()

##############################################################
# Get model names
with open('cv_models.json','r') as f:
    js = json.load(f)

##############################################################
# Get the runtime of the models

op_dict_mxnet = defaultdict(list)


# cum_dist_of_new_ops = []
for j, model_name in enumerate(js["models"]):


    print("running on model :: " + model_name)
    block = run_mxnet_on_tvm.get_mxnet_model(model_name)
    graph, lib, params = run_mxnet_on_tvm.compile(x, block, target, input_opt_level)
    op_time_dict = run_mxnet_on_tvm.execute(graph, lib, params, ctx, x, synset)

    for op in op_time_dict:
        op_dict_mxnet[op].append(op_time_dict[op])

    # run_mxnet_on_tvm.save_and_check_load(block, model_name)

with open('logs/final_runtime_dict.json', 'w') as f:
    json.dump(op_dict_mxnet, f, indent=2)

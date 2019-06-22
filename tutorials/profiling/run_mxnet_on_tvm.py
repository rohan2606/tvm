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

For us to begin with, mxnet module is required to be installed.

A quick solution is

.. code-block:: bash

    pip install mxnet --user

or please refer to offical installation guide.
https://mxnet.incubator.apache.org/versions/master/install/index.html


**Code Modified for op-profiling by : **
Rohan Mukherjee <https://github.com/rohan2606>


"""

# some standard imports
import mxnet as mx
import tvm
import tvm.relay as relay
import numpy as np

from mxnet.gluon.model_zoo.vision import get_model




######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
def get_mxnet_model(name):
    block = get_model(name, pretrained=True)
    return block


######################################################################
# Compile the Graph
# -----------------
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.
# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
def compile(x, block,  target, input_opt_level):
    shape_dict = {'data': x.shape}
    mod, params = relay.frontend.from_mxnet(block, shape_dict)
    ## we want a probability so add a softmax operator
    func = mod[mod.entry_func]
    func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

    with relay.build_config(opt_level=input_opt_level):
        graph, lib, params = relay.build(func, target, params=params)

    return graph, lib, params


######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now, we would like to reproduce the same forward computation using TVM.
def execute(graph, lib, params, ctx, x, synset):
    from tvm.contrib import graph_runtime
    from tvm.contrib.debugger import debug_runtime

    dtype = 'float32'
    m = debug_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(x.astype(dtype)))
    m.set_input(**params)
    # execute
    op_time_dict = m.run()
    # get outputs
    tvm_output = m.get_output(0)
    top1 = np.argmax(tvm_output.asnumpy()[0])
    print('TVM prediction top-1:', top1, synset[top1])

    return op_time_dict



def save_and_check_load(block, model_name):
    model_name = "logs/models/" + model_name
    mx_sym, args, auxs = block2symbol(block)
    # usually we would save/load it as checkpoint
    mx.model.save_checkpoint(model_name , 0, mx_sym, args, auxs)
    # there are 'resnet18_v1-0000.params' and 'resnet18_v1-symbol.json' on disk

    ######################################################################
    # for a normal mxnet model, we start from here
    mx_sym, args, auxs = mx.model.load_checkpoint(model_name , 0)
    # now we use the same API to get Relay computation graph


######################################################################
# Use MXNet symbol with pretrained weights
# ----------------------------------------
# MXNet often use `arg_params` and `aux_params` to store network parameters
# separately, here we show how to use these weights with existing API
def block2symbol(block):
    data = mx.sym.Variable('data')
    sym = block(data)
    args = {}
    auxs = {}
    for k, v in block.collect_params().items():
        args[k] = mx.nd.array(v.data().asnumpy())
    return sym, args, auxs

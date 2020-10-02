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
import numpy as np
import time

import tvm
from tvm import relay
import tvm.relay.testing
import tvm.relay.tensorrt
import pytest
from tvm.contrib import graph_runtime
from tvm import te
from tvm.relay import Any


def test_conv2d_dynamic(x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), groups=1, padding=(0, 0), strides=(1, 1), dilation=(1, 1)):
    x = relay.var('x', shape=(1, 32, Any(), 8), dtype='float32')
    kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
    out = relay.nn.conv2d(x, kernel, channels=k_shape[0], kernel_size=k_shape[2:4], groups=groups, padding=padding, strides=strides, dilation=dilation)
    f = relay.Function([x, kernel], out)
    return f, {'x': x_shape, 'kernel': k_shape}, ['kernel']


def test_conv2d_relu_dynamic(x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), groups=1, padding=(0, 0), strides=(1, 1), dilation=(1, 1)):
    x = relay.var('x', shape=(1, 32, Any(), 8), dtype='float32')
    kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
    out = relay.nn.conv2d(x, kernel, channels=k_shape[0], kernel_size=k_shape[2:4], groups=groups, padding=padding, strides=strides, dilation=dilation)
    out = relay.nn.relu(out)
    f = relay.Function([x, kernel], out)
    return f, {'x': x_shape, 'kernel': k_shape}, ['kernel']



def test_erf(x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), groups=1, padding=(0, 0), strides=(1, 1), dilation=(1, 1)):
    x = relay.var('x', shape=(1, 32, Any(), 8), dtype='float32')
    kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
    out = relay.erf(x)
    f = relay.Function([x, kernel], out)
    return f, {'x': x_shape, 'kernel': k_shape}, ['kernel']



def test_conv2d_erf(x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), groups=1, padding=(0, 0), strides=(1, 1), dilation=(1, 1)):
    x = relay.var('x', shape=(1, 32, Any(), 8), dtype='float32')
    kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
    out = relay.nn.conv2d(x, kernel, channels=k_shape[0], kernel_size=k_shape[2:4], groups=groups, padding=padding, strides=strides, dilation=dilation)
    out = relay.erf(out)
    f = relay.Function([x, kernel], out)
    return f, {'x': x_shape, 'kernel': k_shape}, ['kernel']


def test_max(x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), groups=1, padding=(0, 0), strides=(1, 1), dilation=(1, 1)):
    x = relay.var('x', shape=(1, 32, Any(), 8), dtype='float32')
    kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
    # out = relay.nn.conv2d(x, kernel, channels=k_shape[0], kernel_size=k_shape[2:4], groups=groups, padding=padding, strides=strides, dilation=dilation)
    out = relay.max(x)
    f = relay.Function([x, kernel], out)
    return f, {'x': x_shape, 'kernel': k_shape}, ['kernel']

def test_free_var(x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), groups=1, padding=(0, 0), strides=(1, 1), dilation=(1, 1)):
    # x = relay.var('x', shape=(1, 32, Any(), 8), dtype='float32')
    # kernel = relay.var('kernel', shape=(k_shape), dtype='float32')
    # out = relay.nn.conv2d(x, kernel, channels=k_shape[0], kernel_size=k_shape[2:4], groups=groups, padding=padding, strides=strides, dilation=dilation)
    out = relay.var("shared_bound", shape=(1,), dtype="float32")
    f = relay.Function([out], None)
    return f, {}, []



def run_and_verify(config):
    f, input_shapes, is_param = config
    params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype(np.float32) for x in is_param}
    input_dict = {k: np.random.uniform(-1, 1, v).astype(np.float32) for k, v in input_shapes.items() if
                  k not in is_param}

    # Run TRT
    mod = tvm.IRModule()
    print("Relay Graph")
    mod['main'] = f
    print(mod)
    mod = relay.tensorrt.EnableTrt(mod, params)
    print("Final Relay Graph")
    print(mod)

    # with relay.build_config(opt_level=3):
    #     ex_before = relay.create_executor("vm", mod=mod, ctx=tvm.gpu(0), target='cuda')
    #     results = ex_before.evaluate()(**input_dict, **params)

    # mod = tvm.IRModule()
    # mod['main'] = f
    # with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
    #     ex_after = relay.create_executor("vm", mod=mod, ctx=tvm.cpu(0), target="llvm")
    #     ref_results = ex_after.evaluate()(**input_dict, **params)

    # tvm.testing.assert_allclose(
    #     results.asnumpy(), ref_results.asnumpy(), rtol=1e-3, atol=1e-3
    # )



if __name__ == '__main__':
    # run_and_verify(test_conv2d_dynamic())
    # run_and_verify(test_conv2d_relu_dynamic())
    # run_and_verify(test_erf())
    # run_and_verify(test_conv2d_erf())
    run_and_verify(test_free_var())

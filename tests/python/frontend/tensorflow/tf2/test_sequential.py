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
# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
# pylint: disable=import-outside-toplevel, redefined-builtin
"""TF2 to relay converter test: x+1 as a very basic example"""

import tempfile
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants \
    import convert_variables_to_constants_v2

from test_common import sorted_same, run
from tf_helper import run_tf_code

class TensorArrayStackLayer(tf.keras.layers.Layer):
    def __init__(self, name="tensor_array_test"):
        super().__init__(name=name)

    def call(self, inputs):
        inputs = tf.squeeze(inputs)
        outputs = tf.TensorArray(tf.float32, size=inputs.shape[0],
            infer_shape=False, element_shape=inputs.shape[1:])
        outputs = outputs.unstack(inputs)

        return outputs.stack()

class TensorArrayReadLayer(tf.keras.layers.Layer):
    def __init__(self, name="tensor_array_test"):
        super().__init__(name=name)

    def call(self, inputs):
        inputs = tf.squeeze(inputs)
        outputs = tf.TensorArray(tf.float32, size=inputs.shape[0],
            infer_shape=False, element_shape=inputs.shape[1:])
        for i in range(inputs.shape[0]):
            outputs = outputs.write(i, inputs[i,:])

        return outputs.read(0)

class SequentialModels:
    def __init__(self):
        self.input_shape = (1, 28, 28)
        self.shape = (self.input_shape[1], self.input_shape[2])

    def get_model(self, key, freeze=True):
        if key == "dense":
            return self.dense_model()
        elif key == "mnist":
            return self.mnist_model()
        elif key == "conv2d":
            return self.conv2d_model()
        elif key == "max_pool":
            return self.maxpool_model()
        elif key == "batch_norm":
            return self.batchnorm_model()
        elif key == "tensorlist_stack":
            return self.tensorlist_model(stack=True)
        elif key == "tensorlist_read":
            return self.tensorlist_model(stack=False)
        else:
            assert False

    def save_and_reload(self, model):
        with tempfile.TemporaryDirectory() as model_path:
            tf.saved_model.save(model, model_path)
            loaded = tf.saved_model.load(model_path)
            func = loaded.signatures["serving_default"]
            frozen_func = convert_variables_to_constants_v2(func)
        return frozen_func

    def freeze(self, model):
        # Convert the seq model to tf function
        f = tf.function(model).get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(f)
        return frozen_func

    def reshape_model(self, freeze=True):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=self.shape),
            ]
        )
        if freeze:
            exp_ops = ['Placeholder', 'Const', 'Reshape', 'Identity']
        else:
            exp_ops = ['Placeholder', 'Const', 'Reshape']
        return model, exp_ops

    def tensorlist_model(self, stack=False):
        self.input_shape = (1, 3, 32)
        self.shape = (3, 32)
        if stack:
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=self.shape, batch_size=1),
                    TensorArrayStackLayer()
                ]
            )
            exp_ops = ['Placeholder', 'TensorListStack', 'Identity', 'Squeeze', 'Const', 'TensorListFromTensor']
            return model, exp_ops
        else:
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=self.shape, batch_size=1),
                    TensorArrayReadLayer()
                ]
            )
            exp_ops = ['TensorListSetItem', 'Const', 'Placeholder', 'StridedSlice', 'Squeeze', 'Identity', 'TensorListReserve', 'TensorListGetItem']
            return model, exp_ops

    def dense_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=self.shape),
                tf.keras.layers.Dense(4)
            ]
        )
        exp_ops = ['Placeholder', 'Const', 'Reshape', 'MatMul', 'BiasAdd', 'Identity']
        return model, exp_ops

    def mnist_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10)
        ])

        exp_ops = ['Placeholder', 'Const', 'Reshape', 'MatMul', 'BiasAdd', 'Identity', 'Relu']
        return model, exp_ops

    def conv2d_model(self):
        filters = 16
        kernel = (3, 3)
        self.input_shape = (1, 32, 32, 3)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3), batch_size=1),
            tf.keras.layers.Conv2D(filters, kernel)
        ])

        exp_ops = ['Placeholder', 'Const', 'Conv2D', 'BiasAdd', 'Identity']
        return model, exp_ops

    def maxpool_model(self):
        self.input_shape = (1, 32, 32, 3)
        model = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), input_shape=self.input_shape[1:])
        ])

        exp_ops = ['Placeholder', 'Identity', 'MaxPool']
        return model, exp_ops


    def batchnorm_model(self):

        self.input_shape = (1, 32, 32, 3)
        model = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2,2), input_shape=self.input_shape[1:]),
            tf.keras.layers.BatchNormalization()
        ])

        exp_ops = ['Placeholder', 'Identity', 'Const', 'FusedBatchNormV3', 'MaxPool']
        return model, exp_ops


def test_sequential_model(key, mode="freeze"):
    def check_op_coverage(gdef, expected_ops):
        gdef_ops = list(set([n.op for n in gdef.node]))
        if 'NoOp' in gdef_ops:
            gdef_ops.remove('NoOp')
        assert sorted_same(expected_ops, gdef_ops)

    def get_input(shape):
        input = np.random.uniform(0,1,shape).astype(dtype='float32')
        return input

    def function_params_seq_nofreeze():
        g = tf.Graph()
        model = SequentialModels()
        with g.as_default():
            f, expected_ops = model.get_model(key, freeze=False)
        input = get_input(model.input_shape)
        output = run_tf_code(f, input)
        gdef = g.as_graph_def()
        check_op_coverage(gdef, expected_ops)
        return gdef, input, output

    def function_params_seq_freeze():
        model = SequentialModels()
        seq, expected_ops = model.get_model(key, freeze=True)
        f = model.freeze(seq)
        input = get_input(model.input_shape)
        output = run_tf_code(f, input)
        gdef = f.graph.as_graph_def()
        check_op_coverage(gdef, expected_ops)
        return gdef, input, output

    def model_graph():
        model = SequentialModels()
        seq, expected_ops = model.get_model(key)
        input = get_input(model.input_shape)
        f = model.save_and_reload(seq)
        output = run_tf_code(f, input)
        gdef = f.graph.as_graph_def(add_shapes=True)
        check_op_coverage(gdef, expected_ops)
        return gdef, input, output

    if mode == "freeze":
        run(function_params_seq_freeze)
    elif mode == "nofreeze":
        run(function_params_seq_nofreeze)
    elif mode == "reload":
        run(model_graph)
    else:
        assert False

if __name__ == "__main__":

    for mode in ["freeze", "reload"]:
        test_sequential_model("dense", mode=mode)
        test_sequential_model("mnist", mode=mode)
        test_sequential_model("max_pool", mode=mode)
        test_sequential_model("batch_norm", mode=mode)
        test_sequential_model("conv2d", mode=mode)

        test_sequential_model("tensorlist_stack", mode=mode)
        test_sequential_model("tensorlist_read", mode=mode)
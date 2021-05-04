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
# pylint: disable=invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
"""Tensorflow2.x to relay converter ops and helper"""
import warnings

import tvm
from .common import get_relay_op
from .. import expr as _expr
from .. import op as _op

import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes

from tvm.topi.utils import get_const_tuple
from tvm.relay.prelude import Prelude, StaticTensorArrayOps, get_tensor_array_shape, TensorArrayOps
from ..ty import Any, TensorType
from .common import infer_value as _infer_value
from .tensorflow_ops import _infer_type_with_prelude, _get_more_static_shape


def _need_prelude_for_shape_inference(op):
    return "TensorList" in op or "TensorArray" in op

def _tensorlist_reserve():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr.get("element_dtype").name
        elem_shape = _infer_value(inputs[0], params, prelude.mod)
        elem_shape = tuple(elem_shape.asnumpy().astype("int32").flatten())
        assert -1 not in elem_shape, "TensorList size and element shape must be static"

        static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, elem_shape)
        static_tensor_array_ops.register()
        tensor_array_constructor = static_tensor_array_ops.get_global_var("tensor_array")
        tensor_array = tensor_array_constructor(inputs[1])
        return tensor_array

    return _impl

def _tensorlist_set_item():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr.get("element_dtype").name
        input_ta = inputs[0]
        input_ta_shape = get_tensor_array_shape(input_ta, dtype_str, prelude)
        input_t_shape = _infer_type_with_prelude(inputs[2], prelude).shape

        static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_ta_shape)
        static_tensor_array_ops.register()
        tensor_func = static_tensor_array_ops.get_ctor("tensor_constructor")
        v = tensor_func(inputs[2])
        # Write tensor with more static shape
        actual_shape = _get_more_static_shape(input_t_shape, input_ta_shape)
        if actual_shape != input_t_shape:
            new_shape = []
            num_any_dim = 0
            for dim in actual_shape:
                if not isinstance(dim, int):
                    num_any_dim += 1
                new_shape.append(dim if isinstance(dim, int) else -1)
            if num_any_dim <= 1:
                v = tensor_func(_op.reshape(inputs[2], new_shape))
        write_func = prelude.get_global_var_static(
            "tensor_array_write", dtype_str, input_ta_shape
        )
        out = write_func(input_ta, inputs[1], v)
        return out

    return _impl

def _tensorlist_get_item():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr["element_dtype"].name
        input_shape = get_tensor_array_shape(inputs[0], dtype_str, prelude)

        static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_shape)
        static_tensor_array_ops.register()
        read_func = static_tensor_array_ops.get_global_var("tensor_array_read")
        out_tensor = read_func(inputs[0], _op.take(inputs[1], tvm.relay.const(0)))
        get_data_func = static_tensor_array_ops.get_global_var("tensor_get_data")
        out = get_data_func(out_tensor)
        return out

    return _impl

def _tensorlist_stack():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr["element_dtype"].name
        input_ta_shape = get_tensor_array_shape(inputs[0], dtype_str, prelude)
        assert input_ta_shape is not None

        static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_ta_shape)
        static_tensor_array_ops.register()
        stack_func = prelude.get_global_var_static("tensor_array_stack", dtype_str, input_ta_shape)
        out_tensor = stack_func(inputs[0])
        out_shape = (Any(),) + input_ta_shape
        static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, out_shape)
        static_tensor_array_ops.register()
        get_data_func = prelude.get_global_var_static("tensor_get_data", dtype_str, out_shape)
        out = get_data_func(out_tensor)

        return out

    return _impl

def _tensorlist_from_tensor():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr["element_dtype"].name
        input_ta_shape = _infer_type_with_prelude(inputs[0], prelude).shape
        assert input_ta_shape is not None

        static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_ta_shape)
        static_tensor_array_ops.register()
        unstack_func = prelude.get_global_var_static("tensor_array_unstack", dtype_str, input_ta_shape)
        out = unstack_func(inputs[0])
        return out

    return _impl

_convert_map = {
    "TensorListFromTensor": _tensorlist_from_tensor(),
    "TensorListGetItem": _tensorlist_get_item(),
    "TensorListReserve": _tensorlist_reserve(),
    "TensorListSetItem": _tensorlist_set_item(),
    "TensorListStack": _tensorlist_stack(),
}

def set_span(sym, node_name):
    span = tvm.relay.Span(tvm.relay.SourceName(node_name), 0, 0, 0, 0)
    if isinstance(sym, _expr.Call):
        sym = _expr.Call(sym.op, sym.args, sym.attrs, sym.type_args, span)
    elif isinstance(sym, _expr.TupleWrapper):
        tuple_value = sym.tuple_value
        if isinstance(tuple_value, _expr.Call):
            tuple_value = _expr.Call(
                tuple_value.op, tuple_value.args, tuple_value.attrs, tuple_value.type_args, span
            )
            sym = _expr.TupleWrapper(tuple_value, sym.size)
    return sym

def _get_const_value(node):
    return node.attr["value"].tensor

def convert_const_node(node):
    """convert tf const node into relay const or var
    """
    tensor_value = _get_const_value(node)
    np_array = tensor_util.MakeNdarray(tensor_value)

    if np_array.dtype == np.dtype(object):
        assert False # not tested, maybe tf string type?

    if len(np_array.shape) == 0:
        param = None
        sym = [tvm.relay.const(np_array, np_array.dtype)]
    else:
        param = tvm.nd.array(np_array)
        sym = [
            _expr.var(node.name, shape=param.shape, dtype=param.dtype)
        ]

    return sym, param


def get_attr(buf):
    """convert value of a node attribute. node attribute is part of a node in a graph.

        // tensorflow/core/framework/attr_value.proto
        message AttrValue {
            oneof value {
                bytes s = 2;                 // "string"
                int64 i = 3;                 // "int"
                float f = 4;                 // "float"
                bool b = 5;                  // "bool"
                DataType type = 6;           // "type"
                TensorShapeProto shape = 7;  // "shape"
                TensorProto tensor = 8;      // "tensor"
                ListValue list = 1;          // any "list(...)"            }
        }        

    Parameters
    ----------
    buf: attrvalue protobuf.  <class 'tensorflow.core.framework.attr_value_pb2.AttrValue'>

    Returns
    -------
    The value of the attr, as a Python object.

    """
    fields = ["s", "i", "f", "b", "type", "shape", "tensor", "func"]

    x = buf

    ret = []

    if not x.WhichOneof("value"):
        assert False # not yet tested; why would there be empty attribute value in a node def?

    if x.HasField("list"):
        for f in fields:
            if getattr(x.list, f):
                if f == "type":
                    ret += [dtypes.as_dtype(x) for x in list(getattr(x.list, f))]
                else:
                    ret += list(getattr(x.list, f))
    else:
        for f in fields:
            if x.HasField(f):
                if f == "type":
                    ret = dtypes.as_dtype(getattr(x, f))
                else:
                    ret = getattr(x, f)
    return ret
    
def parse_attr(attr_proto):
    """Convert node attributes (a serialized map of key-value pairs) in a node to a dict

    Parameters
    ----------
    attr_proto: <class 'google.protobuf.pyext._message.MessageMapContainer'>
    attributes of a tf node

    protobuf message format:
        // tensorflow/core/framework/node_def.proto
        message NodeDef {
            map<string, AttrValue> attr = 5;
        }

    Returns
    -------
    Dict {string: python object}


    Examples
    --------
    attributes in following node converted to {'_user_specified_name': b'x', 'dtype': tf.float32 }

        node {
        name: "x"
        op: "Placeholder"
        attr {
            key: "_user_specified_name"
            value {
            s: "x"
            }
        }
        attr {
            key: "dtype"
            value {
            type: DT_FLOAT
            }
        }


    """
    attrs = {}
    for key, value in attr_proto.items():
        attrs[key] = get_attr(value)

    return attrs

def convert_place_holder(shape, node, in_type=None):
    """ convert tf place holder into relay var. 
    
    Examples
    --------    
    a tf place holder with name "x" is converted to [Var(x, ty=TensorType([], float32))]

    """

    if shape and node.name in shape:
        input_shape = list(shape[node.name])
        assert False # not yet tested
    else:
        input_shape = tensor_util.TensorShapeProtoToList(
            node.attr["shape"].shape
        )
        for idx, dim in enumerate(input_shape):
            if dim < 0:
                input_shape[idx] = Any()
    attr = parse_attr(node.attr)
    if in_type is not None:
        sym = [
            _expr.var(
                node.name, type_annotation=in_type
            )
        ]
    else:
        sym = [
            _expr.var(
                node.name, shape=input_shape, dtype=attr["dtype"].name
            )
        ]
    return input_shape, sym



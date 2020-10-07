import os

import time
import numpy as np
import tvm
import tvm.relay.tensorrt
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay.testing.temp_op_attr import TempOpAttr
import argparse
import ctypes

from tvm.runtime.vm import VirtualMachine
from tvm.runtime import container as _container


def run_and_verify(config):
    f, input_shapes, is_param = config
    params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype(np.float32) for x in is_param}
    input_dict = {k: np.random.uniform(-1, 1, v).astype(np.float32) for k, v in input_shapes.items() if
                  k not in is_param}

    # Run VM with TRT
    mod = tvm.IRModule()
    mod['main'] = f
    mod = relay.tensorrt.EnableTrt(mod, params)
    with relay.build_config(opt_level=3):
        ex_before = relay.create_executor("vm", mod=mod, ctx=tvm.gpu(0), target="cuda")
        results = ex_before.evaluate()(**input_dict, **params)

    # Run VM without TRT
    mod = tvm.IRModule()
    mod['main'] = f
    with relay.build_config(opt_level=3):
        ex_after = relay.create_executor("vm", mod=mod, ctx=tvm.gpu(0), target="cuda")
        ref_results = ex_after.evaluate()(**input_dict, **params)

    if isinstance(results, _container.ADT):
        for result, ref_result in zip(results, ref_results):
            tvm.testing.assert_allclose(
                result.asnumpy(), ref_result.asnumpy(), rtol=1e-3, atol=1e-3
            )
    else:
        tvm.testing.assert_allclose(
            results.asnumpy(), ref_results.asnumpy(), rtol=1e-3, atol=1e-3
        )
    print(results)
    print(ref_results)

def test_if_node():
  data = relay.var('data', shape=(1, 32),  dtype='float32')
  eq1 = relay.var('e1', shape=(), dtype='float32')
  eq2 = relay.var('e2', shape=(), dtype='float32')
  eq = relay.equal(eq1, eq2)

  true_branch = relay.tanh(data)
  false_branch = relay.sigmoid(data)
  ife = relay.If(eq, true_branch, false_branch)
  out = relay.erf(ife)
  func = relay.Function([data, eq1, eq2], out)
  # mod = tvm.IRModule.from_expr(func)
  return func, {'data': (1, 32), 'e1': (), 'e2': () }, []



def test_recursive():
    mod = tvm.IRModule()

    x = relay.var("x", shape=(2,), dtype="float32")
    i = relay.var("i", shape=(), dtype="int32")
    s = relay.var("s", shape=(2,), dtype="float32")

    cond = i < relay.const(10, dtype="int32")

    loop = relay.var("while_loop")
    sb = relay.scope_builder.ScopeBuilder()
    with sb.if_scope(cond):
        ii = i + relay.const(1, dtype="int32")
        ss = s + x
        sb.ret(loop(ii, ss))
    with sb.else_scope():
        sb.ret(s)
    func = relay.Function([i, s], sb.get())

    ret = relay.Let(
        loop, func, loop(relay.const(0, dtype="int32"), relay.zeros_like(x))
    )
    mod["main"] = relay.Function([x], ret)

    # mod = tvm.IRModule.from_expr(func)
    return func, {'x': (2,), 'i': (), 's': (2,)}, []
    # params = dict()
    # print(mod)
    # mod = relay.tensorrt.EnableTrt(mod, params)
    # print(mod)


if __name__ == '__main__':
  # run_and_verify(test_if_node())
  run_and_verify(test_recursive())
  # print("Finish")


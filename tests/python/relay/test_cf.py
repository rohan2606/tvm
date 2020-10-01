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

def test_if_node():
  data = relay.var('data', shape=(1, 32))
  eq1 = relay.var('e1', shape=[], dtype='int32')
  eq2 = relay.var('e2', shape=[], dtype='int32')
  eq = relay.equal(eq1, eq2)

  true_branch = relay.tanh(data)
  false_branch = relay.sigmoid(data)
  ife = relay.If(eq, true_branch, false_branch)
  out = relay.erf(ife)
  func = relay.Function([data, eq1, eq2], out)
  mod = tvm.IRModule.from_expr(func)
  params = dict()
  print(mod)
  mod = relay.tensorrt.EnableTrt(mod, params)
  print(mod)

def test_recursive():
    mod = tvm.IRModule()

    x = relay.var("x", shape=(2,))
    i = relay.var("i", shape=(), dtype="int32")
    s = relay.var("s", shape=(2,))
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

    params = dict()
    print(mod)
    mod = relay.tensorrt.EnableTrt(mod, params)
    print(mod)


if __name__ == '__main__':
  # test_if_node()
  test_recursive()
  # print("Finish")


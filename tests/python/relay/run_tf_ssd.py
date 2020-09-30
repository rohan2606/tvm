import os

import tensorflow as tf
import time
import numpy as np
import tvm
import tvm.relay.tensorrt
from tvm import relay
from tvm.relay.frontend.tensorflow_parser import TFParser
from tvm.contrib import graph_runtime
from tvm.relay.testing.temp_op_attr import TempOpAttr
import argparse
import ctypes

from tvm.runtime.vm import VirtualMachine

def save_mod_params(mod, params, name):
  saved_mod = tvm.runtime._ffi_node_api.SaveJSON(mod)
  with open(name, "w") as of:
    of.write(saved_mod)
  with open(name+".params", "wb") as fo:
    fo.write(relay.save_param_dict(params))
  # return

def load_mod_params(name):
  with open("{}".format(name), "r") as inf:
    saved = inf.read()
  mod = tvm.runtime._ffi_node_api.LoadJSON(saved)
  params = relay.load_param_dict(bytearray(open("{}.params".format(name), "rb").read()))
  return mod, params

def benchmark(model='saved_model',
              input_name='image_tensor',
              name='ssd_mn',
              input_shape=(1, 512, 512, 3)):

  parser = TFParser(model)
  graph_def = parser.parse()

  if not os.path.exists(name):
      mod, params = relay.frontend.from_tensorflow(graph_def, shape={input_name: input_shape})
      save_mod_params(mod, params, name)
  else:
      mod, params = load_mod_params(name)


  mod = relay.tensorrt.EnableTrt(mod, params)
  print(mod)

  # seq = tvm.transform.Sequential([
  #     relay.transform.RemoveUnusedFunctions(),
  #     relay.transform.ConvertLayout({'nn.conv2d': ['NCHW', 'default'],
  #                                    'nn.conv3d': ['NCDHW', 'default'],
  #                                    'qnn.conv2d': ['NCHW', 'default'],
  #                                    'qnn.conv3d': ['NCDHW', 'default']})
  # ])
  # with tvm.transform.PassContext(opt_level=3):
  #   mod = seq(mod)

  # print(mod)


  # dtype = 'float32'
  # i_data = np.random.uniform(-1, 1, input_shape).astype(dtype)
  # feed_dict = {input_name: i_data}
  #
  # with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
  #   vm_exec = relay.vm.compile(mod, "llvm", params=params)
  # ctx = tvm.cpu()
  # vm = VirtualMachine(vm_exec, ctx)
  # vm.set_input("main", **feed_dict)
  # tvm_res = vm.run()
  #

if __name__ == '__main__':
  path = '/home/ubuntu/pytorch_od/models/ssd_mobilenet/saved_model'
  benchmark(model=path, name='ssd_mn')


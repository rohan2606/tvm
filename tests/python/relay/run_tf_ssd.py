import os

import cv2
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
import tvm.relay.testing.tf as tf_testing

from tvm.runtime.vm import VirtualMachine

from tvm.contrib.download import download


def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x


tf_dtypes = {
    "float32": tf.float32,
    "float16": tf.float16,
    "float64": tf.float64,
    "int32": tf.int32,
    "uint8": tf.uint8,
    "int8": tf.int8,
    "int16": tf.int16,
    "uint16": tf.uint16,
    "int64": tf.int64,
}


def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        return [o.asnumpy()]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f))
        return result
    elif isinstance(o, tvm.relay.backend.interpreter.ConstructorValue):
        if o.constructor.name_hint == "Cons":
            tl = vmobj_to_list(o.fields[1])
            hd = vmobj_to_list(o.fields[0])
            hd.extend(tl)
            return hd
        elif o.constructor.name_hint == "Nil":
            return []
        elif "tensor_nil" in o.constructor.name_hint:
            return [0]
        elif "tensor" in o.constructor.name_hint:
            return [o.fields[0].asnumpy()]
        else:
            raise RuntimeError("Unknown object type: %s" % o.constructor.name_hint)
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def compile(model='saved_model',
            name='ssd_mn',
            input_name='image_tensor',
            input_shape=(1, 512, 512, 3),
            out_names=None,
            use_trt=True,
            ):

  def save_mod_params(mod, params, name):
    saved_mod = tvm.runtime._ffi_node_api.SaveJSON(mod)
    with open(name, "w") as of:
      of.write(saved_mod)
    with open(name + ".params", "wb") as fo:
      fo.write(relay.save_param_dict(params))
    # return

  def load_mod_params(name):
    with open("{}".format(name), "r") as inf:
      saved = inf.read()
    mod = tvm.runtime._ffi_node_api.LoadJSON(saved)
    params = relay.load_param_dict(bytearray(open("{}.params".format(name), "rb").read()))
    return mod, params


  if not os.path.exists(name):
      parser = TFParser(model, outputs=out_names)
      graph_def = parser.parse()

      # print("Processing GraphDefPraram")
      graph_def = tf_testing.ProcessGraphDefParam(graph_def)

      # print(graph_def)

      print("Getting frontend")
      mod, params = relay.frontend.from_tensorflow(graph_def, shape={input_name: input_shape},
                                                   outputs=out_names)
      save_mod_params(mod, params, name)
  else:
      mod, params = load_mod_params(name)

  final_name = name + '_trt' if use_trt else name

  if not os.path.exists(final_name + "_lib.so"):
    if use_trt:
      mod = relay.tensorrt.EnableTrt(mod, params, prune_subgraphs=True)

    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
      vm_exec = relay.vm.compile(mod, "llvm", params=params)
      code, lib = vm_exec.save()

    # save and load the code and lib file.
    lib.export_library(final_name + "_lib.so")
    with open(final_name + "_code.ro", "wb") as fo:
        fo.write(code)

    print("Compilation done")
  else:
    print("Compilation extracted")


def benchmark(
        name='ssd_mn',
        i_data=None):

  print("Running now")

  loaded_lib = tvm.runtime.load_module(name + "_lib.so")
  loaded_code = bytearray(open(name + "_code.ro", "rb").read())
  # deserialize.
  ctx = tvm.cpu()
  des_exec = tvm.runtime.vm.Executable.load_exec(loaded_code, loaded_lib)

  des_vm = tvm.runtime.vm.VirtualMachine(des_exec, ctx)

  result = des_vm.invoke("main", i_data)
  return result



def infer_from_tf(path,
        i_data=None):

  parser = TFParser(path)
  graph_def = parser.parse()

  with tf.Session() as sess:
    image_tensor = sess.graph.get_tensor_by_name("image_tensor:0")
    out_node = ["detection_boxes", "detection_scores", "detection_classes"]
    detection_boxes, detection_scores, detection_classes = \
      sess.run(["{}:0".format(oname) for oname in out_node], feed_dict={image_tensor: i_data})

  return detection_boxes, detection_scores, detection_classes

if __name__ == '__main__':
  path = '/home/ubuntu/pytorch_od/models/ssd_mobilenet/saved_model'
  out_node = ["detection_boxes", "detection_scores", "detection_classes"]

  # compile(model=path, name='ssd_mn', use_trt=True, out_names=out_node)
  compile(model=os.path.join(path, 'saved_model.pb'),
          name='ssd_mn',
          use_trt=True,
          out_names=out_node)

  np.random.seed(100)
  # Changes by Zhi
  # i_data = np.random.uniform(0.0, 255.0, size=(1, 512, 512, 3)).astype("uint8")

  ######################################################################
  # # Download a test image and pre-process
  # # -------------------------------------
  # in_size = 300
  #
  # img_path = "test_street_small.jpg"
  # img_url = (
  #     "https://raw.githubusercontent.com/dmlc/web-data/" "master/gluoncv/detection/street_small.jpg"
  # )
  # download(img_url, img_path)
  #
  # img = cv2.imread(img_path).astype("float32")
  # img = cv2.resize(img, (in_size, in_size))
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # img = np.transpose(img / 255.0, [2, 0, 1])
  # img = np.expand_dims(img, axis=0)
  # i_data = np.transpose(img, [0, 3, 1, 2])
  # i_data = i_data.astype(np.uint8)

  # print(i_data)
  ##
  time_trt = 0.
  time_vm = 0.
  for i in range(100):
    ##
    i_data = np.random.uniform(0.0, 255.0, size=(1, 512, 512, 3)).astype("uint8")

    t1_start = time.perf_counter_ns()
    res_trt = benchmark(name='ssd_mn_trt', i_data=i_data)
    t1_end = time.perf_counter_ns()
    time_trt += (t1_end - t1_start)
    # ##
    t2_start = time.perf_counter_ns()
    res_vm = benchmark(name='ssd_mn', i_data=i_data)
    t2_end = time.perf_counter_ns()
    time_vm += (t2_end - t2_start)

    # print(vmobj_to_list(res_trt))
    # print(vmobj_to_list(res_vm))

    # predictions = infer_from_tf( os.path.join(path, 'saved_model.pb') , i_data=i_data)


    # for result, ref_result in zip(vmobj_to_list(res_vm), predictions):
    for result, ref_result in zip(vmobj_to_list(res_vm), vmobj_to_list(res_trt)):
      tvm.testing.assert_allclose(
        result, ref_result, rtol=1e-3, atol=1e-3
      )
    # print(predictions)


    # for m, n in zip(res_trt.asnumpy()[0], res_vm.asnumpy()[0]):
    #   assert m == n
    # print(res_trt.asnumpy()[0], res_vm.asnumpy()[0])
    print(time_vm /(np.power(10, 9) * (i+1)), time_trt/(np.power(10, 9) * (i+1)))
  print("Finish")

  # predictions = infer_from_tf( os.path.join(path, 'saved_model.pb') , i_data=i_data)
  # print(predictions)
  # print(res_vm)


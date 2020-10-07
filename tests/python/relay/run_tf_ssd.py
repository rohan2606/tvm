import os

import time
import numpy as np
import tvm
import tvm.relay.tensorrt
from tvm import relay
# from tvm.relay.frontend.tensorflow_parser import TFParser
import tvm.relay.testing.tf as tf_testing
from tvm.runtime.vm import VirtualMachine
from tvm.testing import vmobj_to_list


def dont_support(attrs, args):
  return False


tvm.ir.register_op_attr('nn.conv2d', 'target.tensorrt', level=11, value=dont_support)

# Add more operators here  - go 1 by 1
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

  from tvm.relay.frontend.tensorflow_parser import TFParser

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


  print("Running now VM+TRT")

  loaded_lib = tvm.runtime.load_module(name + "_lib.so")
  loaded_code = bytearray(open(name + "_code.ro", "rb").read())
  # deserialize.
  ctx = tvm.cpu()
  des_exec = tvm.runtime.vm.Executable.load_exec(loaded_code, loaded_lib)

  des_vm = tvm.runtime.vm.VirtualMachine(des_exec, ctx)

  result = des_vm.invoke("main", i_data)
  print("Running done VM+TRT")
  return result



def infer_from_tf(path,
        i_data=None):
  import tensorflow as tf
  from tvm.relay.frontend.tensorflow_parser import TFParser

  print("Running now TF")

  parser = TFParser(path)
  graph_def = parser.parse()

  with tf.Session() as sess:
    image_tensor = sess.graph.get_tensor_by_name("image_tensor:0")
    out_node = ["detection_boxes", "detection_scores", "detection_classes"]
    detection_boxes, detection_scores, detection_classes = \
      sess.run(["{}:0".format(oname) for oname in out_node], feed_dict={image_tensor: i_data})

  print("Running done TF")
  return detection_boxes, detection_scores, detection_classes

if __name__ == '__main__':
  path = '/home/ubuntu/pytorch_od/models/ssd_mobilenet/saved_model'
  out_node = ["detection_boxes", "detection_scores", "detection_classes"]

  compile(model=os.path.join(path, 'saved_model.pb'),
          name='ssd_mn',
          use_trt=True,
          out_names=out_node)

  np.random.seed(101)

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

    print(vmobj_to_list(res_trt))
    print(vmobj_to_list(res_vm))

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


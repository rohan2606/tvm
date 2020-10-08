import os
import time
import numpy as np
# import tensorflow as tf
import tvm
import tvm.relay.tensorrt
from tvm import relay
from tvm.relay.frontend.tensorflow_parser import TFParser
import tvm.relay.testing.tf as tf_testing
from tvm.runtime.vm import VirtualMachine


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


# def dont_support(attrs, args):
#   return False
# tvm.ir.register_op_attr('add', 'target.tensorrt', level=11, value=dont_support)


def compile_tvm(model='saved_model',
                name='ssd_mn',
                input_name='image_tensor',
                input_shape=(1, 512, 512, 3),
                out_names=None,
                use_trt=True,
                ):

    mod_param_path = name
    lib_path = name + '_trt' if use_trt else name

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

    if not os.path.exists(mod_param_path):
        parser = TFParser(model, outputs=out_names)
        graph_def = parser.parse()
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        mod, params = relay.frontend.from_tensorflow(graph_def, shape={input_name: input_shape},
                                                     outputs=out_names)
        save_mod_params(mod, params, name)
    else:
        mod, params = load_mod_params(name)


    if not os.path.exists(lib_path + "_lib.so"):
        if use_trt:
            mod = relay.tensorrt.EnableTrt(mod, params, prune_subgraphs=True)

        with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
            vm_exec = relay.vm.compile(mod, "llvm", params=params)
            code, lib = vm_exec.save()

        # save and load the code and lib file.
        lib.export_library(lib_path + "_lib.so")
        with open(lib_path + "_code.ro", "wb") as fo:
            fo.write(code)
        loaded_lib = tvm.runtime.load_module(lib_path + "_lib.so")
        loaded_code = bytearray(open(lib_path + "_code.ro", "rb").read())
        print("Compilation done")
    else:
        loaded_lib = tvm.runtime.load_module(lib_path + "_lib.so")
        loaded_code = bytearray(open(lib_path + "_code.ro", "rb").read())
        print("Compilation extracted")
    return loaded_lib, loaded_code


def infer_from_tf(path,
                  i_data=None):
    import tensorflow as tf
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


def run(lib_vm, code_vm, lib_trt, code_trt, num_runs=100):

    def run_once(des_vm, i_data):
        t_start = time.perf_counter()
        res_trt = des_vm.invoke("main", i_data)
        t_end = time.perf_counter()
        t = t_end - t_start
        return res_trt, t

    # setup
    np.random.seed(int(time.time()))
    time_trt, time_vm = 0., 0.
    des_exec_vm = tvm.runtime.vm.Executable.load_exec(code_vm, lib_vm)
    des_exec_trt = tvm.runtime.vm.Executable.load_exec(code_trt, lib_trt)
    ctx = tvm.cpu()
    des_vm = tvm.runtime.vm.VirtualMachine(des_exec_vm, ctx)
    des_vm_trt = tvm.runtime.vm.VirtualMachine(des_exec_trt, ctx)

    ## Run
    for i in range(num_runs):
        ## Generate data
        i_data = np.random.uniform(0.0, 255.0, size=(1, 512, 512, 3)).astype("uint8")

        ## Run MN with TRT
        res_trt, t = run_once(des_vm, i_data)
        time_trt += t

        ### Run MN with VM
        res_vm, t = run_once(des_vm_trt, i_data)
        time_vm += t

        ## Run TF
        # predictions = infer_from_tf( os.path.join(path, 'saved_model.pb') , i_data=i_data)

        ## Compare accuracies
        for result, ref_result in zip(vmobj_to_list(res_vm), vmobj_to_list(res_trt)):
            tvm.testing.assert_allclose(
                result, ref_result, rtol=1e-3, atol=1e-3
            )

        ## Print average runtime
        print("Time to complete with VM {} and time to complete with TRT {}".format(time_vm / ((i + 1)),
                                                                                    time_trt / ((i + 1))))

        # predictions = infer_from_tf( os.path.join(path, 'saved_model.pb') , i_data=i_data)
        # print(predictions)
        # print(res_vm)


if __name__ == '__main__':
    path = '/home/ubuntu/tf_models/saved_model'
    out_node = ["detection_boxes", "detection_scores", "detection_classes"]

    lib_vm, code_vm = compile_tvm(model=os.path.join(path, 'saved_model.pb'),
                              name='ssd_mn',
                              use_trt=False,
                              out_names=out_node)

    lib_trt, code_trt = compile_tvm(model=os.path.join(path, 'saved_model.pb'),
                              name='ssd_mn',
                              use_trt=True,
                              out_names=out_node)

    run(lib_vm, code_vm, lib_trt, code_trt )
    print("Finish")

import mxnet as mx 
def gpu_device(gpu_number=0):
    try:
        _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(gpu_number))
    except mx.MXNetError:
        return None
    return mx.gpu(gpu_number)


gpu_device = gpu_device()
if not gpu_device:
    print('No GPU device found!')
else:
    print('GPU device available')

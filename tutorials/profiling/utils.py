from PIL import Image
import tvm
from tvm.contrib.download import download_testdata
from matplotlib import pyplot as plt
import numpy as np

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def get_image():
    img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
    img_name = 'cat.png'
    synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                          '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                          '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                          'imagenet1000_clsid_to_human.txt'])
    synset_name = 'imagenet1000_clsid_to_human.txt'
    img_path = download_testdata(img_url, 'cat.png', module='data')
    synset_path = download_testdata(synset_url, synset_name, module='data')
    with open(synset_path) as f:
        synset = eval(f.read())
    image = Image.open(img_path).resize((224, 224))
    # plt.imshow(image)
    # plt.show()
    image = transform_image(image)
    return image, synset


def get_opt_params(sys_argv):
    if len(sys_argv) != 3:
       print('Please specify which target to use:\'cuda\'/\'llvm\' and which opt to use:\'fused\'/\'unfused\'')
       sys.exit()

    if sys_argv[1] == 'cuda':
       target = 'cuda'
       ctx = tvm.gpu(0)
    elif sys_argv[1] == 'llvm':
       target = 'llvm'
       ctx = tvm.cpu(0)
    else:
       print('wrong target! Exiting...')
       sys.exit()


    if sys_argv[2] == 'fused':
       input_opt_level = 3
    elif sys_argv[2] == 'unfused':
       input_opt_level = 0
    else:
       print('wrong opt level! Exiting...')
       sys.exit()

    return target, ctx, input_opt_level

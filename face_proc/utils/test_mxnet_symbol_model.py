import mxnet as mx
import numpy as np
from collections import namedtuple


def get_device(device):
    if device == 'cpu':
        ctx = mx.cpu()
    elif device == 'cuda':
        ctx = mx.gpu(0)
    else:
        ctx = None
    return ctx


def get_model(model_path, ctx): 
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    return mod


def get_image(img_rgb):
    img = mx.nd.array(img_rgb)
    img = mx.image.imresize(img, 224, 224)
    img = img.transpose((2, 0, 1))
    """
    # pytorch onnx model
    img = mx.nd.array(img, dtype=np.float32)
    img = img / 255
    """
    """
    # changsha model
    img = mx.nd.array(img, dtype=np.float32)
    img = img / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for c in range(3):
        img[c] = (img[c] - mean[c]) / std[c]
    """
    img = img.expand_dims(axis=0) # batchify
    return img


def predict_image(model_path, img_rgb, device):
    ctx = get_device(device)
    model = get_model(model_path, ctx)

    img = get_image(img_rgb).copyto(ctx)
    Batch = namedtuple('Batch', ['data'])
    model.forward(Batch([img]))

    prob = model.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]

    pred = a[0]
    prob0 = float(prob[a[0]])

    return [pred, prob0]

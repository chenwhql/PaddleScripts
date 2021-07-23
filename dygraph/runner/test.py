import paddle.fluid as fluid
import paddlehub as hub
import cv2
import numpy as np
from PIL import Image

DATA_DIM = 224
img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def process_image(img):
    img = Image.fromarray(img[:, :, ::-1])
    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=DATA_DIM, center=True)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std
    return img

if __name__ == '__main__':
    module = hub.Module(name='resnet50_vd_imagenet_ssld')
    with fluid.dygraph.guard():
        img = cv2.imread('mask.jpg')
        data = process_image(img)[np.newaxis, :, :, :]
        
        # 模型预测
        layer = fluid.dygraph.StaticModelRunner(module.default_pretrained_model_path)
        result = layer(data)
        print(result)
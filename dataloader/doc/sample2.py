'''
Example in dynamic graph mode.
'''
import numpy as np

import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.distributed as dist

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 1

USE_GPU = False # whether to use GPU

def _get_random_images_and_labels(image_shape, label_shape):
        image = np.random.random(size=image_shape).astype('float32')
        label = np.random.random(size=label_shape).astype('int64')
        return image, label

def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = _get_random_images_and_labels(
                [BATCH_SIZE, IMAGE_SIZE], [BATCH_SIZE, CLASS_NUM])
            yield batch_image, batch_label

def random_batch_reader():
    return __reader__

class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    @paddle.jit.to_static
    def forward(self, x):
        return self._linear(x)

# set device
paddle.set_device('gpu' if USE_GPU else 'cpu')

# create network
layer = LinearNet()
dp_layer = paddle.DataParallel(layer)
loss_fn = nn.CrossEntropyLoss()
adam = opt.Adam(learning_rate=0.001, parameters=dp_layer.parameters())

# create data loader
loader = paddle.io.DataLoader.from_generator(capacity=5)
loader.set_batch_generator(random_batch_reader())

for epoch_id in range(EPOCH_NUM):
    for batch_id, (image, label) in enumerate(loader()):
        out = layer(image)
        loss = loss_fn(out, label)

        loss.backward()

        adam.step()
        adam.clear_grad()
        print("Epoch {} batch {}: loss = {}".format(
            epoch_id, batch_id, np.mean(loss.numpy())))
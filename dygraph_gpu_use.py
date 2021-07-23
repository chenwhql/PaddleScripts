import numpy as np
import paddle.fluid as fluid
import os
from paddle.fluid.dygraph import Linear

BATCH_SIZE = 32
BATCH_NUM = 20

def multi_gpus_used():
    res = os.popen("nvidia-smi --query-gpu=memory.used --format=csv").read()
    lines = res.splitlines()
    memory_use = []
    for line in lines[1:]:
        units = line.split(' ')
        memory_use.append(int(units[0]))
    for mem in memory_use:
        if mem <= 10:
            return False
    return True

def random_batch_reader():
    def _get_random_images_and_labels(image_shape, label_shape):
        image = np.random.random(size=image_shape).astype('float32')
        label = np.random.random(size=label_shape).astype('int64')
        return image, label

    def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = _get_random_images_and_labels(
                [BATCH_SIZE, 784], [BATCH_SIZE, 1])
            yield batch_image, batch_label

    return __reader__

class LinearNet(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(LinearNet, self).__init__()
        self._linear = Linear(in_size, out_size)

    def forward(self, x):
        return self._linear(x)

# enable dygraph mode
place = fluid.CUDAPlace(0)
fluid.enable_dygraph(place) 

# 1. train & save model.
# create network
net = LinearNet(784, 1)
adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
# create data loader
train_loader = fluid.io.DataLoader.from_generator(capacity=5)
train_loader.set_batch_generator(random_batch_reader(), places=place)
# train
for data in train_loader():
    img, label = data
    label.stop_gradient = True

    cost = net(img)

    error = multi_gpus_used()

    loss = fluid.layers.cross_entropy(cost, label)
    avg_loss = fluid.layers.mean(loss)

    avg_loss.backward()
    adam.minimize(avg_loss)
    net.clear_gradients()

# print whether use multi gpu
print(error)
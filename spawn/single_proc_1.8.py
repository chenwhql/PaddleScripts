import numpy as np
import paddle
import paddle.fluid as fluid

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 10

IMAGE_SIZE = 784
CLASS_NUM = 10

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

class LinearNet(fluid.dygraph.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = fluid.dygraph.Linear(IMAGE_SIZE, CLASS_NUM)

    def forward(self, x):
        return self._linear(x)

def train():
    with fluid.dygraph.guard(fluid.CUDAPlace(2)):
        # create network
        layer = LinearNet()
        adam = fluid.optimizer.Adam(learning_rate=0.001, parameter_list=layer.parameters())

        # print(core._get_device_properties(dist.ParallelEnv().device_id))

        # create data loader
        # loader = paddle.io.DataLoader.from_generator(capacity=5, use_multiprocess=True)
        loader = paddle.io.DataLoader.from_generator(capacity=5)
        loader.set_batch_generator(random_batch_reader())

        for epoch_id in range(EPOCH_NUM):
            for batch_id, (image, label) in enumerate(loader()):
                out = layer(image)
                out.backward()
                adam.minimize(out)
                # adam.clear_grad()
                print("Epoch {} batch {}: loss = {}".format(
                    epoch_id, batch_id, np.mean(out.numpy())))

if __name__ == '__main__':
    train()

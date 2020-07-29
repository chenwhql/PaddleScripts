import sys
import unittest
import numpy as np
import paddle.fluid as fluid

batch_size = 8
batch_num = 1
epoch_num = 1
capacity = 2

def get_random_images_and_labels(image_shape, label_shape):
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return image, label

def batch_generator_creator(batch_size, batch_num):
    def __reader__():
        for _ in range(batch_num):
            batch_image, batch_label = get_random_images_and_labels(
                [batch_size, 784], [batch_size, 1])
            yield batch_image, batch_label

    return __reader__

def test_batch_genarator():
    with fluid.dygraph.guard():
        loader = fluid.io.DataLoader.from_generator(
            capacity=capacity, use_multiprocess=True)
        loader.set_batch_generator(
            batch_generator_creator(batch_size, batch_num),
            places=fluid.CUDAPlace(0))
        for _ in range(epoch_num):
            for image, _ in loader():
                fluid.layers.relu(image)
        print("run success.")

if __name__ == '__main__':
    test_batch_genarator()
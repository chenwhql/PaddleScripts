import paddle.fluid as fluid
import numpy as np

BATCH_NUM = 10 
BATCH_SIZE = 16
EPOCH_NUM = 4

def get_random_images_and_labels(image_shape, label_shape):
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return image, label

def simple_net(image, label):
    fc_tmp = fluid.layers.fc(image, size=CLASS_NUM)
    cross_entropy = fluid.layers.softmax_with_cross_entropy(image, label)
    loss = fluid.layers.reduce_mean(cross_entropy)
    sgd = fluid.optimizer.SGD(learning_rate=1e-3)
    sgd.minimize(loss)
    return loss

def sample_list_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM): 
            sample_list = []
            for _ in range(BATCH_SIZE):
                image, label = get_random_images_and_labels([784], [1])
                sample_list.append([image, label])

            yield sample_list

    return __reader__

image = fluid.data(name='image', shape=[None, 784], dtype='float32')
label = fluid.data(name='label', shape=[None, 1], dtype='int64')

# define reader
reader = fluid.io.PyReader(
        feed_list=[image, label], capacity=6, iterable=True, return_list=False)
reader.decorate_sample_list_generator(sample_list_generator_creator(), places=fluid.CPUPlace())

# train
it = reader.__iter__()
it.next()  # 拿一个batch的数据
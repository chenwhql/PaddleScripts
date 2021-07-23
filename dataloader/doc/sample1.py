'''
Example in static graph mode
'''
import numpy as np

import paddle
import paddle.static as static
import paddle.nn.functional as F


BATCH_NUM = 10
BATCH_SIZE = 16
EPOCH_NUM = 4

CLASS_NUM = 10

ITERABLE = True # whether the created DataLoader object is iterable
USE_GPU = False # whether to use GPU

DATA_FORMAT = 'batch_generator' # data format of data source user provides

paddle.enable_static()

def simple_net(image, label):
    fc_tmp = static.nn.fc(image, size=CLASS_NUM)
    cross_entropy = F.softmax_with_cross_entropy(image, label)
    loss = paddle.mean(cross_entropy)
    sgd = paddle.optimizer.SGD(learning_rate=1e-3)
    sgd.minimize(loss)
    return loss

def get_random_images_and_labels(image_shape, label_shape):
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return image, label

# If the data generator yields one sample each time,
# use DataLoader.set_sample_generator to set the data source.
def sample_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM * BATCH_SIZE):
            image, label = get_random_images_and_labels([784], [1])
            yield image, label

    return __reader__

# If the data generator yield list of samples each time,
# use DataLoader.set_sample_list_generator to set the data source.
def sample_list_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM):
            sample_list = []
            for _ in range(BATCH_SIZE):
                image, label = get_random_images_and_labels([784], [1])
                sample_list.append([image, label])

            yield sample_list

    return __reader__

# If the data generator yields a batch each time,
# use DataLoader.set_batch_generator to set the data source.
def batch_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = get_random_images_and_labels([BATCH_SIZE, 784], [BATCH_SIZE, 1])
            yield batch_image, batch_label

    return __reader__

# If DataLoader is iterable, use for loop to train the network
def train_iterable(exe, prog, loss, loader):
    for _ in range(EPOCH_NUM):
        for data in loader():
            exe.run(prog, feed=data, fetch_list=[loss])

# If DataLoader is not iterable, use start() and reset() method to control the process
def train_non_iterable(exe, prog, loss, loader):
    for _ in range(EPOCH_NUM):
        loader.start() # call DataLoader.start() before each epoch starts
        try:
            while True:
                exe.run(prog, fetch_list=[loss])
        except paddle.core.EOFException:
            loader.reset() # call DataLoader.reset() after catching EOFException

def set_data_source(loader, places):
    if DATA_FORMAT == 'sample_generator':
        loader.set_sample_generator(sample_generator_creator(), batch_size=BATCH_SIZE, drop_last=True, places=places)
    elif DATA_FORMAT == 'sample_list_generator':
        loader.set_sample_list_generator(sample_list_generator_creator(), places=places)
    elif DATA_FORMAT == 'batch_generator':
        loader.set_batch_generator(batch_generator_creator(), places=places)
    else:
        raise ValueError('Unsupported data format')

image = static.data(name='image', shape=[None, 784], dtype='float32')
label = static.data(name='label', shape=[None, 1], dtype='int64')

# Define DataLoader
loader = paddle.io.DataLoader.from_generator(feed_list=[image, label], capacity=16, iterable=ITERABLE)

# Define network
loss = simple_net(image, label)

# Set data source of DataLoader
#
# If DataLoader is iterable, places must be given and the number of places must be the same with device number.
#  - If you are using GPU, call `paddle.static.cuda_places()` to get all GPU places.
#  - If you are using CPU, call `paddle.static.cpu_places()` to get all CPU places.
#
# If DataLoader is not iterable, places can be None.
places = static.cuda_places() if USE_GPU else static.cpu_places()
set_data_source(loader, places)

exe = static.Executor(places[0])
exe.run(static.default_startup_program())

prog = static.CompiledProgram(static.default_main_program()).with_data_parallel(loss_name=loss.name)

if loader.iterable:
    train_iterable(exe, prog, loss, loader)
else:
    train_non_iterable(exe, prog, loss, loader)
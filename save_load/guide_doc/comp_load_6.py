import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.optimizer as opt

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

image = fluid.data(name='image', shape=[None, 784], dtype='float32')
label = fluid.data(name='label', shape=[None, 1], dtype='int64')
pred = fluid.layers.fc(input=image, size=10, act='softmax')
loss = fluid.layers.cross_entropy(input=pred, label=label)
avg_loss = fluid.layers.mean(loss)

optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_loss)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# create data loader
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
loader = paddle.io.DataLoader(dataset,
    feed_list=[image, label],
    places=place,
    batch_size=BATCH_SIZE, 
    shuffle=True,
    drop_last=True,
    num_workers=2)

# train model
for data in loader():
    exe.run(
        fluid.default_main_program(),
        feed=data, 
        fetch_list=[avg_loss])

# save by fluid.io.save_params
model_path = "fc.example.model.save_params_with_filename"
fluid.io.save_params(exe, model_path, filename="__params__")

# load 
import os
params_file_path = os.path.join(model_path, "__params__")
var_list = fluid.default_main_program().all_parameters()
state_dict = paddle.io.load_program_state(params_file_path, var_list)
print(state_dict)
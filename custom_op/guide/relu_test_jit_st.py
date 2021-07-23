import numpy as np
import paddle
import paddle.nn as nn
import paddle.static as static
from paddle.utils.cpp_extension import load

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

# jit compile custom op
custom_ops = load(
    name="custom_jit_ops",
    sources=["relu_cuda.cc", "relu_cuda.cu"])

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

# switch static mode
paddle.enable_static()
paddle.set_device("gpu")

# create network
image  = static.data(shape=[None, IMAGE_SIZE], name='image', dtype='float32')
label = static.data(shape=[None, 1], name='label', dtype='int64')

tmp1 = static.nn.fc(x=image, size=100)
tmp_out = custom_ops.custom_relu(tmp1)
tmp2 = static.nn.fc(x=tmp_out, size=CLASS_NUM)
out = custom_ops.custom_relu(tmp2)

loss = nn.functional.cross_entropy(out, label)
opt = paddle.optimizer.SGD(learning_rate=0.01)
opt.minimize(loss)

# create data loader
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
loader = paddle.io.DataLoader(dataset,
    feed_list=[image, label],
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# prepare
exe = static.Executor()
exe.run(static.default_startup_program())

# train
for epoch_id in range(EPOCH_NUM):
    for batch_id, (image_data, label_data) in enumerate(loader()):
        loss_data = exe.run(static.default_main_program(),
            feed={'image': image_data,
                  'label': label_data},
            fetch_list=[loss])
        print("Epoch {} batch {}: loss = {}".format(
                epoch_id, batch_id, np.mean(loss_data)))

# save inference model
path = "custom_relu_test_static/net"
static.save_inference_model(path, [image], [out], exe)
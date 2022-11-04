import numpy as np
import paddle
import paddle.static as static
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.nn.functional as F

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

paddle.enable_static()

image = static.data(name='image', shape=[None, 784], dtype='float32')
label = static.data(name='label', shape=[None, 1], dtype='int64')
pred = static.nn.fc(x=image, size=10, activation='softmax')
loss = F.cross_entropy(input=pred, label=label)
avg_loss = paddle.mean(loss)

optimizer = paddle.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_loss)

place = paddle.CPUPlace()
exe = static.Executor(place)
exe.run(static.default_startup_program())

# create data loader
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
loader = paddle.io.DataLoader(dataset,
    feed_list=[image, label],
    places=place,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    return_list=False,
    num_workers=2)

# 1. train and save inference model
for data in loader():
    exe.run(
        static.default_main_program(),
        feed=data,
        fetch_list=[avg_loss])

model_path = "fc.example.model"
paddle.fluid.io.save_inference_model(
    model_path, ["image"], [pred], exe)

# 2. load model

# enable dygraph mode
paddle.disable_static(place)

# load
fc = paddle.jit.load(model_path)

# inference
fc.eval()
x = paddle.randn([1, IMAGE_SIZE], 'float32')
pred = fc(x)

# fine-tune
fc.train()
loss_fn = nn.CrossEntropyLoss()
adam = opt.Adam(learning_rate=0.001, parameters=fc.parameters())
loader = paddle.io.DataLoader(dataset,
    places=place,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)
for epoch_id in range(EPOCH_NUM):
    for batch_id, (image, label) in enumerate(loader()):
        out = fc(image)
        loss = loss_fn(out, label)
        loss.backward()
        adam.step()
        adam.clear_grad()
        print("Epoch {} batch {}: loss = {}".format(
            epoch_id, batch_id, np.mean(loss.numpy())))
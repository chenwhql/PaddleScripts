import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.framework as framework

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 10

IMAGE_SIZE = 784
CLASS_NUM = 10

# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float64')
        label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    def forward(self, x):
        return self._linear(x)

def train():
    paddle.set_default_dtype("float64")
    # create network
    layer = LinearNet()
    loss_fn = nn.CrossEntropyLoss()
    adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

    # create data loader
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    loader = paddle.io.DataLoader(dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=1)

    # train
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)

            loss.backward()

            adam.step()
            adam.clear_grad()

            print("Epoch {} batch {}: loss = {}".format(
                epoch_id, batch_id, np.mean(loss.numpy())))

if __name__ == '__main__':
    train()
import paddle
from paddle.vision.models import resnet50
from paddle.vision.datasets import Cifar10
from paddle.vision.transforms import Compose

EPOCH_NUM = 10
BATCH_SIZE = 32

def train():
    paddle.set_device('gpu')

    model = resnet50()
    paddle.summary(model, (1, 3, 32, 32))

    transform = Compose([
        paddle.vision.transforms.Transpose(),
        paddle.vision.transforms.Normalize(0, 255.),
    ])
    cifar10 = Cifar10(mode='train', transform=transform)

    loader = paddle.io.DataLoader(cifar10,
                                  shuffle=True,
                                  batch_size=BATCH_SIZE,
                                  num_workers=10)
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(loader()):
            out = model(data[0])
            out = paddle.mean(out)
            if batch_id % 10 == 0:
                print("Epoch {}: batch {}, out {}".format(epoch, batch_id, out.numpy()))

if __name__ == '__main__':
    train()
import paddle
import paddle.vision.transforms as transforms

# will segment fault
num_workers = 0

# OK
# num_workers = 1
trainset = paddle.vision.datasets.MNIST(mode='test', transform=transforms.ToTensor())

trainloader = paddle.io.DataLoader(trainset, batch_size=32, num_workers=num_workers, shuffle=True)

for epoch_id in range(3):
    print("start iter")
    for batch_idx, data in enumerate(trainloader):
        print("epoch {}, batch {}".format(epoch_id, batch_idx))
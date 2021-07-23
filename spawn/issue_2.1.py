import os

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader
from paddle.vision import transforms

normalize = transforms.Normalize(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(), transforms.Transpose(),
    normalize
])

def train():
    # 设置支持多卡训练
    dist.init_parallel_env()

    train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
    batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=32, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)

    model = paddle.vision.mobilenet_v2(num_classes=10)
    # 设置支持多卡训练
    model = paddle.DataParallel(model)

    # 设置优化方法
    optimizer = paddle.optimizer.SGD(parameters=model.parameters(),
                                     learning_rate=0.1,
                                     weight_decay=5e-4)

    # 获取损失函数
    loss = paddle.nn.CrossEntropyLoss()
    # 开始训练
    for epoch in range(10):
        for batch_id, (img, label) in enumerate(train_loader()):
            output = model(img)
            # 计算损失值
            los = loss(output, label)
            los.backward()
            if dist.get_rank() == 0:
                print("Epoch {}: batch_id {}, loss {}".format(epoch, batch_id, los))
            optimizer.step()
            optimizer.clear_grad()

if __name__ == '__main__':
    dist.spawn(train)
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.static import InputSpec

def train():
    device = paddle.set_device('cpu') # or 'gpu'

    net = nn.Sequential(
        nn.Flatten(1),
        nn.Linear(784, 200),
        nn.Tanh(),
        nn.Linear(200, 10))

    # inputs and labels are not required for dynamic graph.
    input = InputSpec([None, 784], 'float32', 'x')
    label = InputSpec([None, 1], 'int64', 'label')

    model = paddle.Model(net, input, label)
    optim = paddle.optimizer.SGD(learning_rate=1e-3,
        parameters=model.parameters())
    model.prepare(optim,
                  paddle.nn.CrossEntropyLoss(),
                  paddle.metric.Accuracy())

    data = paddle.vision.datasets.MNIST(mode='train')
    model.fit(data, epochs=2, batch_size=32, verbose=1)

if __name__ == '__main__':
    dist.spawn(train)
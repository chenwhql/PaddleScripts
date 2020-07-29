import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph import declarative

BATCH_SIZE = 32
BATCH_NUM = 20

def random_batch_reader():
    def _get_random_images_and_labels(image_shape, label_shape):
        image = np.random.random(size=image_shape).astype('float32')
        label = np.random.random(size=label_shape).astype('int64')
        return image, label

    def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = _get_random_images_and_labels(
                [BATCH_SIZE, 784], [BATCH_SIZE, 1])
            yield batch_image, batch_label

    return __reader__

class LinearNet(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(LinearNet, self).__init__()
        self._linear = Linear(in_size, out_size)

    @declarative
    def forward(self, x):
        return self._linear(x)

# enable dygraph mode
fluid.enable_dygraph() 

# 1. train & save model.
# create network
net = LinearNet(784, 1)
adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
# create data loader
train_loader = fluid.io.DataLoader.from_generator(capacity=5)
train_loader.set_batch_generator(random_batch_reader())
# train
for data in train_loader():
    img, label = data
    label.stop_gradient = True

    cost = net(img)

    loss = fluid.layers.cross_entropy(cost, label)
    avg_loss = fluid.layers.mean(loss)

    avg_loss.backward()
    adam.minimize(avg_loss)
    net.clear_gradients()

model_path = "linear.example.model"
fluid.dygraph.jit.save(
    layer=net,
    model_path=model_path,
    input_spec=[img])

# 2. load model & inference
# load model
infer_net = fluid.dygraph.jit.load(model_path)
# inference
x = fluid.dygraph.to_variable(np.random.random((1, 784)).astype('float32'))
pred = infer_net(x)

# 3. load model & fine-tune
# load model
train_net = fluid.dygraph.jit.load(model_path)
train_net.train()
adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=train_net.parameters())
# create data loader
train_loader = fluid.io.DataLoader.from_generator(capacity=5)
train_loader.set_batch_generator(random_batch_reader())
# fine-tune
for data in train_loader():
    img, label = data
    label.stop_gradient = True

    cost = train_net(img)

    loss = fluid.layers.cross_entropy(cost, label)
    avg_loss = fluid.layers.mean(loss)

    avg_loss.backward()
    adam.minimize(avg_loss)
    train_net.clear_gradients()
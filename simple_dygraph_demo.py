import paddle.fluid as fluid
import numpy as np

BATCH_NUM = 10
BATCH_SIZE = 16

class MyLayer(fluid.dygraph.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.linear = fluid.dygraph.nn.Linear(784, 10)

    def forward(self, inputs, label=None):
        x = self.linear(inputs)
        if label is not None:
            loss = fluid.layers.cross_entropy(x, label)
            avg_loss = fluid.layers.mean(loss)
            return x, avg_loss
        else:
            return x

# 伪数据生成函数，服务于下述三种不同的生成器
def get_random_images_and_labels(image_shape, label_shape):
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return image, label


# 每次生成一个Sample List，使用set_sample_list_generator配置数据源
def sample_list_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM):
            sample_list = []
            for _ in range(BATCH_SIZE):
                image, label = get_random_images_and_labels([784], [1])
                sample_list.append([image, label])

            yield sample_list

    return __reader__

place = fluid.CPUPlace() # 或者 fluid.CUDAPlace(0)
with fluid.dygraph.guard(place):

    # 创建执行的网络对象
    my_layer = MyLayer()

    # 添加优化器
    adam = fluid.optimizer.AdamOptimizer(
        learning_rate=0.001, parameter_list=my_layer.parameters())

    # 配置DataLoader
    train_loader = fluid.io.DataLoader.from_generator(capacity=10)
    train_loader.set_sample_list_generator(sample_list_generator_creator(), places=place)
    
    # 执行训练/预测
    for data in train_loader():
        image, label = data

        # 执行前向
        x, avg_loss = my_layer(image, label)

        # 执行反向
        avg_loss.backward()

        # 梯度更新
        adam.minimize(avg_loss)
        my_layer.clear_gradients()

        print(x.numpy())
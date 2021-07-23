import paddle
import numpy as np
 
paddle.set_device("CPU")
np.random.seed(42)
 
class Optimization_ex1(paddle.nn.Layer):
    def __init__(self, shape, param_attr=paddle.nn.initializer.Uniform(low=-5., high=5.), dtype='float64'):
        super(Optimization_ex1, self).__init__()
 
        # 初始化一个长度为 theta_size的可学习参数列表，并用 [-5, 5] 的均匀分布来填充初始值
        self.theta0 = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        self.theta1 = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        self.A = paddle.to_tensor(np.random.random((4, 4)) + np.random.random((4, 4)) * 1j)
        self.B = paddle.to_tensor(np.random.random((4, 4)) + np.random.random((4, 4)) * 1j, stop_gradient=False)
 
    # 定义损失函数和前向传播机制
    def forward(self, mode=1):
        jj = paddle.to_tensor(np.array([1j]))
        if mode == 1:
            # 这里就是一步写完
            loss = paddle.sum(self.A + (self.theta0 + self.theta1 * jj)) * (
                paddle.sum(self.A + (self.theta0 + self.theta1 * jj)).conj())
            return loss.real()
        elif mode == 2:
            # 这里就是分成两步，使用一个中间变量
            self.theta = self.theta0 + self.theta1 * jj
            loss = paddle.sum(self.A + self.theta) * (paddle.sum(self.A + self.theta).conj())
            return loss.real()
        elif mode == 3:
            # 这里就是直接生成一个复矩阵作为参数，然后进行优化。
            loss = paddle.sum(self.A + self.B) * (paddle.sum(self.A + self.B).conj())
            return loss.real()
        else:
            raise NotImplementedError
 
 
if __name__ == "__main__":
    # 超参数设置
    theta_size = [4, 4]
    ITR = 100  # 设置迭代次数
    LR = 0.1  # 设置学习速率
    # 记录中间优化结果
    loss_list = []
    parameter_list = []
 
    # 定义网络维度
    myLayer = Optimization_ex1(theta_size)
 
    # 一般来说，我们利用Adam优化器来获得相对好的收敛，当然你可以改成SGD或者是RMSprop.
    optimizer = paddle.fluid.optimizer.AdamOptimizer(learning_rate=LR, parameter_list=myLayer.parameters())
 
    # 优化循环
    for itr in range(ITR):
        # 向前传播计算损失函数
 
        loss = myLayer(3)  # 当这里的参数为1的时候，可以反传梯度并进行计算，当这里的参数为2的时候，则会报错
        # 而具体细看代码的逻辑则会发现，其计算方式是一样的，只不过是分成两步写和在一行代码中写完的区别
        # 更进一步，这里也可以设置参数为3,对于那种计算方式，也会产生报错
        # 这里我们希望3种方法都可以正常运行。
 
        # 在动态图机制下，反向传播优化损失函数
        loss.backward()
        optimizer.minimize(loss)
        myLayer.clear_gradients()
        print(loss.numpy())
        # 记录学习曲线
        loss_list.append(loss.numpy()[0])
        parameter_list.append(myLayer.parameters()[0].numpy())
 
    print('损失函数的最小值是: ', loss_list[-1])
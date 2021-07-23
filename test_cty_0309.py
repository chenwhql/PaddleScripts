import paddle
import numpy as np

paddle.set_device("CPU")
# np.random.seed(47)


class Optimization_ex1(paddle.nn.Layer):
    def __init__(self, shape, param_attr=paddle.nn.initializer.Uniform(low=-5., high=5.), dtype='float64'):
        # 在paddle2.0.1版本下，当dtype为float64时会报错，为float32时不报错
        super(Optimization_ex1, self).__init__()

        # 初始化一个长度为 theta_size的可学习参数列表，并用 [-5, 5] 的均匀分布来填充初始值
        self.theta0 = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        self.theta1 = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        self.A = paddle.to_tensor(np.random.random((4, 4)).astype(dtype) + np.random.random((4, 4)).astype(dtype) * 1j)
        self.B = paddle.to_tensor(np.random.random((4, 4)).astype(dtype) + np.random.random((4, 4)).astype(dtype) * 1j)
        # , stop_gradient=False

    # 定义损失函数和前向传播机制
    def forward(self, problem=1):
        jj = paddle.to_tensor(np.array([1j], dtype=np.complex64))
        # tr = paddle.trace(self.A)
        # _A = paddle.abs(self.A)
        self.A = paddle.abs(self.A)
        self.theta2 = self.theta0 + self.theta1 * jj
        if problem == 1:
            # matmul不支持不同dtype之间的运算
            loss = paddle.sum(paddle.matmul(self.theta0, self.A))
            return loss
        elif problem == 2:
            # kron在复数情况下不支持梯度反传
            tmp_theta = paddle.kron(self.theta0, self.theta2)
            tmp_A = paddle.kron(self.A, self.B)
            loss = paddle.sum(paddle.matmul(tmp_theta, tmp_A))
            return loss.real()
        elif problem == 3:
            # loss有时会越来越大
            loss = paddle.sum(paddle.matmul(self.theta2, self.A))
            # print("loss: ", loss)
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
    optimizer = paddle.optimizer.Adam(learning_rate=LR, parameters=myLayer.parameters())

    # 优化循环
    for itr in range(ITR):
        # 向前传播计算损失函数
        # 这里的参数可以为1、2、3，分别代表问题1、2、3
        loss = myLayer(1)
        # loss = myLayer(2)
        # loss = myLayer(3)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        # print(itr, loss.numpy())
        # 记录学习曲线
        loss_list.append(loss.numpy()[0])
        parameter_list.append(myLayer.parameters()[0].numpy())

    print('损失函数的最小值是: ', loss_list[-1])

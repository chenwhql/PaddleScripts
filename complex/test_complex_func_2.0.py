import paddle
import numpy as np

# 超参数设置
theta_size = [4, 4]
ITR = 200       # 设置迭代次数
LR = 0.5        # 设置学习速率
paddle.set_device('cpu')

class Optimization_ex1(paddle.nn.Layer):
    def __init__(self, shape, param_attr=paddle.nn.initializer.Uniform(low=-5., high=5.), dtype='float32'):
        super(Optimization_ex1, self).__init__()
        
        # 初始化一个长度为 theta_size的可学习参数列表，并用 [-5, 5] 的均匀分布来填充初始值
        self.theta = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        self.A = paddle.to_tensor(np.random.random((4, 4)).astype(dtype) + np.random.randn(4, 4).astype(dtype)*1j)  #这里如果是复数会出错。

    # 定义损失函数和前向传播机制
    def forward(self):
        loss = paddle.add(self.theta, self.A)
        return loss.real()
    
# 记录中间优化结果
loss_list = []
parameter_list = []

# 定义网络维度
myLayer = Optimization_ex1(theta_size)
print(myLayer.parameters())

# 一般来说，我们利用Adam优化器来获得相对好的收敛，当然你可以改成SGD或者是RMSprop.
optimizer = paddle.optimizer.Adam(learning_rate=LR, parameters=myLayer.parameters())    

# 优化循环
for itr in range(ITR):
    
    # 向前传播计算损失函数
    loss = myLayer()
    
    # 在动态图机制下，反向传播优化损失函数
    loss.backward()

    optimizer.step()
    # optimizer.clear_grad()
    
    # 记录学习曲线
    loss_list.append(loss.numpy()[0])
    parameter_list.append(myLayer.parameters()[0].numpy())
    
print('损失函数的最小值是: ', loss_list[-1])

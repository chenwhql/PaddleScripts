import paddle
import paddle.incubate.complex as cpx
import numpy as np

theta_size = [4, 4]
ITR = 20
LR = 0.5
paddle.set_device('cpu')

class Optimization_ex1(paddle.nn.Layer):
    def __init__(self, shape, param_attr=paddle.nn.initializer.Uniform(low=-5., high=5.), dtype='float'):
        super(Optimization_ex1, self).__init__()
        
        self.theta = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        self.A = paddle.to_tensor(np.random.randn(4, 4) + np.random.randn(4, 4)*1j)

    def forward(self):
        loss = cpx.elementwise_add(self.theta, self.A)
        return loss
    
loss_list = []
parameter_list = []

myLayer = Optimization_ex1(theta_size)

optimizer = paddle.optimizer.Adam(learning_rate = LR, parameters = myLayer.parameters())    

for itr in range(ITR):
    
    loss = myLayer().real()
    
    loss.backward()

    optimizer.step()
    optimizer.clear_grad()
    
    loss_list.append(loss.numpy()[0])
    parameter_list.append(myLayer.parameters()[0].numpy())
    
print('loss: ', loss_list[-1])

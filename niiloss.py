import paddle
import torch
import numpy as np
import paddle.fluid as fluid
import time 

x_data = np.array([[1, 2], [3, 4]]).astype("float32")
y_data = np.array([[2, 1], [2, 4]]).astype("float32")
z_data = np.array([[1, -1], [-1, -1]]).astype("float32")
np.random.seed(90)
x_data = np.random.random(size=(10, 3)).astype(np.float32)
y_data = np.random.randint(0, 5, size=(10,)).astype(np.int64)
z_data = np.random.randint(0, 3, size=(3,)).astype(np.float32)
# x_data = np.random.random([5, 100, 10, 10]).astype("float64")
# y_data = np.random.randint(0, 100, size=(5, 10, 10)).astype(np.int64)
print("input: {}".format(x_data))
print("label: {}".format(y_data))


# paddle.disable_static(paddle.CPUPlace())
paddle.disable_static(paddle.CUDAPlace(0))
x = paddle.to_variable(x_data)
x.stop_gradient = False
y = paddle.to_variable(y_data)
z = paddle.to_variable(z_data)
# nll_loss = paddle.nn.NLLLoss(weight=z, reduction='mean')
nll_loss = paddle.nn.NLLLoss(reduction='mean')
output = nll_loss(x, y)
output.backward()
print("paddle: {}".format(output.numpy()))
print("paddle.shape: {}".format(output.numpy().shape))
# print("paddle gradient: {}".format(x.gradient()))

try:
    t1=torch.tensor(x_data, requires_grad=True)
    t2=torch.tensor(y_data)
    t=torch.tensor(z_data)
#    nll_loss = torch.nn.NLLLoss(weight=t, reduction='mean')
    nll_loss = torch.nn.NLLLoss(reduction='mean')
    output = nll_loss(t1, t2)
    gradients=torch.ones_like(output)
    output.backward(gradients)
    print("torch {}".format(output))
    print("torch.shape {}".format(output.shape))
#    print("torch grad  {}".format(t1.grad))
except Exception as inst:
     print("torch: {}".format(inst))


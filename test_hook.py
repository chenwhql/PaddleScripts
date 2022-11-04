import paddle
from paddle import nn
from paddle.fluid import core

# def scale_hook(grad):
#     # with paddle.no_grad():
#     grad.stop_gradient = True
#     return grad * 8


class Model(nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(20000, 20000)
        self.fc2 = nn.Linear(20000, 1000)

    def forward(self, x, y):
        w = x + y
        # register hook by lambda function
        o = self.fc1(w)
        o.register_hook(lambda grad: grad * 8)
        # o.register_hook(scale_hook)
        z = self.fc2(o)

        return z


model = Model()
for i in range(10000):
# for i in range(10):

    if i % 100 == 0:
        print('step {}'.format(i))

    x = paddle.rand([1000, 20000])
    y = paddle.rand([1000, 20000])

    o = model(x, y)
    o.backward()

    model.clear_gradients()

    # var_names = core.VarBase._alive_vars()
    # print("num of alive vars: ", len(var_names))
    # print("alias var names: ", var_names)

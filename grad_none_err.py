import paddle


class Model(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = paddle.nn.Conv2D(12, 12, 3)

    def forward(self, x, y):
        x = self.conv(x)
        y = self.conv(y)
        var = y.flatten()

        # x[0, :, 0, 0] *= var
        tmp = x[0, :, 0, 0]
        x = tmp * var
        # x[0, :, 0, 0] = tmp2
        loss = paddle.mean(x)
        print(var)
        var.register_hook(lambda grad: print('var grad', grad.sum()))
        # return loss, var
        return loss


model = Model()

x = paddle.ones([1, 12, 3, 3]).astype("float32")
y = paddle.ones([1, 12, 3, 3]).astype("float32")
# loss, var = model(x, y)
loss = model(x, y)

loss.backward()

# print(var.grad.sum())
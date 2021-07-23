import paddle

x = paddle.zeros([2, 3])
y = paddle.ones([2, 3])
x.stop_gradient = False

z = x + y

opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=[x])

z.backward()
opt.step()

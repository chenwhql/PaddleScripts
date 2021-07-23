import paddle
import paddle.static as static

paddle.enable_static()

img = static.data(name='image', shape=[None, 784])
pred = static.nn.fc(input=img, size=10, act='relu')
loss = paddle.mean(pred)
# Here we use clone before Momentum
test_program = static.default_main_program().clone(for_test=True)
optimizer = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
optimizer.minimize(loss)
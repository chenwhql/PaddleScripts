import paddle
import paddle.static as static

paddle.enable_static()

x1 = static.data(name='x1', shape=[2, 3], dtype='float32')
x2 = static.data(name='x2', shape=[2, 3], dtype='float32')
x3 = static.data(name='x3', shape=[2, 3], dtype='float32')

out1 = paddle.concat(x=[x1, x2, x3], axis=-1)

print(static.default_main_program())
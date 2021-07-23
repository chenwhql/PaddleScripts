import paddle
import paddle.static as static

paddle.enable_static()

data = paddle.randn(shape=[2,3], dtype='float32')
res = paddle.scale(data, scale=2.0, bias=1.0)

print(static.default_main_program())
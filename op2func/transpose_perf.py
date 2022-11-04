import paddle
import yep

paddle.set_device("gpu")
x = paddle.randn([2, 3, 4])

for i in range(100):
    x_transposed = paddle.transpose(x, perm=[1, 0, 2])

yep.start("prof/transpose_dev_new.prof")
for i in range(1000000):
    x_transposed = paddle.transpose(x, perm=[1, 0, 2])
yep.stop()

print("Success")
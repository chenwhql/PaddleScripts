import paddle
import yep

paddle.set_device("cpu")
input = paddle.rand(shape=[4, 5, 6], dtype='float32')
axes = [0, 1, 2]
starts = [-3, 0, 2]
ends = [3, 2, 4]

for i in range(100):
    sliced_1 = paddle.slice(input, axes=axes, starts=starts, ends=ends)

yep.start("prof/slice_dev.prof")
for i in range(500000):
    sliced_1 = paddle.slice(input, axes=axes, starts=starts, ends=ends)
yep.stop()

print("Success")
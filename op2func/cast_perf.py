import paddle
import numpy as np
import yep

paddle.set_device("cpu")
x = paddle.to_tensor([2, 3, 4], 'float64')

yep.start("cast_dev_new.prof")
for i in range(1000000):
    y = paddle.cast(x, 'uint8')
yep.stop()
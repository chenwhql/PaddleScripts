import paddle
import yep

paddle.set_device("gpu")
x = [[2,3,4], [7,8,9]]
x = paddle.to_tensor(x, dtype='float32')

for i in range(100):
    res = paddle.log(x)

yep.start("prof/log_2.2.prof")
for i in range(1000000):
    res = paddle.log(x)
yep.stop()

print("Success")
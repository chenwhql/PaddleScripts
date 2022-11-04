import paddle

a = paddle.rand(shape=[2, 3])
print(a)
a[a > 0.5] = 1
print(a)

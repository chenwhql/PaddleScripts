import paddle

a = 1.5
b = paddle.full([2, 2, 2], True, dtype='bool')
c = a + b

print(c)
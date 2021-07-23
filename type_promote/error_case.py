from __future__ import print_function, division

import paddle

a = paddle.ones([2, 2, 2], dtype='int64')
b = 1.0
print(a + b)

a = paddle.ones([2, 2, 2], dtype='int64')
b = 1.5
print(a + b)

a = paddle.ones([2, 2, 2], dtype='int64')
b = 2
print(a / b)

a = paddle.ones([2, 2, 2], dtype='int64')
b = 0.5
print(a / b)

a = 1
b = paddle.full([2, 2, 2], 2, dtype='int64')
print(a / b)

a = 1.0
b = paddle.full([2, 2, 2], 2, dtype='int64')
print(a / b)

a = paddle.full([2, 2, 2], 2, dtype='int64')
b = 3.0
print(a ** b)
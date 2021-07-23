from __future__ import print_function, division

import paddle

a = paddle.ones([2, 2, 2], dtype='int64')
b = 1.0
print(a + b)

# New:
# Tensor(shape=[2, 2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[[2., 2.],
#          [2., 2.]],

#         [[2., 2.],
#          [2., 2.]]])

a = paddle.ones([2, 2, 2], dtype='int64')
b = 1.5
print(a + b)

# New:
# Tensor(shape=[2, 2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[[2.50000000, 2.50000000],
#          [2.50000000, 2.50000000]],

#         [[2.50000000, 2.50000000],
#          [2.50000000, 2.50000000]]])

a = paddle.ones([2, 2, 2], dtype='int64')
b = 2
print(a / b)

# New:
# Tensor(shape=[2, 2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[[0.50000000, 0.50000000],
#          [0.50000000, 0.50000000]],

#         [[0.50000000, 0.50000000],
#          [0.50000000, 0.50000000]]])

a = paddle.ones([2, 2, 2], dtype='int64')
b = 0.5
print(a / b)

# New:
# Tensor(shape=[2, 2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[[2., 2.],
#          [2., 2.]],

#         [[2., 2.],
#          [2., 2.]]])

a = 1
b = paddle.full([2, 2, 2], 2, dtype='int64')
print(a / b)

# New:
# Tensor(shape=[2, 2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[[0.50000000, 0.50000000],
#          [0.50000000, 0.50000000]],

#         [[0.50000000, 0.50000000],
#          [0.50000000, 0.50000000]]])

a = 1.0
b = paddle.full([2, 2, 2], 2, dtype='int64')
print(a / b)

# New:
# Tensor(shape=[2, 2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[[0.50000000, 0.50000000],
#          [0.50000000, 0.50000000]],

#         [[0.50000000, 0.50000000],
#          [0.50000000, 0.50000000]]])

a = paddle.full([2, 2, 2], 2, dtype='int64')
b = 3.0
print(a ** b)

# New:
# Tensor(shape=[2, 2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[[8., 8.],
#          [8., 8.]],

#         [[8., 8.],
#          [8., 8.]]])
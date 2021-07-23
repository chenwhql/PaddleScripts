from __future__ import print_function, division

import paddle

# a = paddle.ones([2, 2, 2], dtype='int64')
# b = 1.0
# print(a + b)

# Original:
# Tensor(shape=[2, 2, 2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [[[2, 2],
#          [2, 2]],

#         [[2, 2],
#          [2, 2]]])

# a = paddle.ones([2, 2, 2], dtype='int64')
# b = 1.5
# print(a + b)

# Original:
# AssertionError: float value 1.5 cannot convert to integer

# a = paddle.ones([2, 2, 2], dtype='int64')
# b = 2
# print(a / b)

# Original:
# Tensor(shape=[2, 2, 2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [[[0, 0],
#          [0, 0]],

#         [[0, 0],
#          [0, 0]]])

# a = paddle.ones([2, 2, 2], dtype='int64')
# b = 0.5
# print(a / b)

# Original:
# Error: /work/paddle/paddle/fluid/operators/elementwise/elementwise_op_function.cu.h:66 Assertion `b != 0` failed. InvalidArgumentError: Integer division by zero encountered in divide. Please check.

# a = 1
# b = paddle.full([2, 2, 2], 2, dtype='int64')
# print(a / b)

# Original:
# Tensor(shape=[2, 2, 2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [[[0, 0],
#          [0, 0]],

#         [[0, 0],
#          [0, 0]]])

# a = 1.0
# b = paddle.full([2, 2, 2], 2, dtype='int64')
# print(a / b)

# Original:
# Tensor(shape=[2, 2, 2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [[[0, 0],
#          [0, 0]],

#         [[0, 0],
#          [0, 0]]])

a = paddle.full([2, 2, 2], 2, dtype='int64')
b = 3.0
print(a ** b)

# Original:
# Tensor(shape=[2, 2, 2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [[[8, 8],
#          [8, 8]],

#         [[8, 8],
#          [8, 8]]])
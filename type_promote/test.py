import paddle

paddle.set_device('cpu')

a = paddle.ones((2,3), dtype='float32')  # float tensor
b = paddle.ones((2,3), dtype='int32')    # int tensor

c1 = a + 1.5 #  正确
print(c1)
# Tensor(shape=[2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[2.50000000, 2.50000000, 2.50000000],
#         [2.50000000, 2.50000000, 2.50000000]])

c2 = b / 3.5 #  不正确，应该是float
print(c2)
# Tensor(shape=[2, 3], dtype=int32, place=CPUPlace, stop_gradient=True,
#        [[0, 0, 0],
#         [0, 0, 0]])

c3 = b / 3   #  正确
print(c3)
# Tensor(shape=[2, 3], dtype=int32, place=CPUPlace, stop_gradient=True,
#        [[0, 0, 0],
#         [0, 0, 0]])

c5 = a + b   # 正确
print(c5)
# Tensor(shape=[2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[2., 2., 2.],
#         [2., 2., 2.]])

c6 = b + a   # 不正确，应该是float
print(c6)
# Tensor(shape=[2, 3], dtype=int32, place=CPUPlace, stop_gradient=True,
#        [[2, 2, 2],
#         [2, 2, 2]])
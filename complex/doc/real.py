import paddle

input = paddle.to_tensor(
    [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])
# Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
#        [[(1+6j), (2+5j), (3+4j)],
#         [(4+3j), (5+2j), (6+1j)]])

real_res = paddle.real(input)
# Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[1., 2., 3.],
#         [4., 5., 6.]])

real_t = input.real()
# Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[1., 2., 3.],
#         [4., 5., 6.]])
import paddle

input = paddle.to_tensor(
    [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])
# Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
#        [[(1+6j), (2+5j), (3+4j)],
#         [(4+3j), (5+2j), (6+1j)]])

imag_res = paddle.imag(input)
# Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[6., 5., 4.],
#         [3., 2., 1.]])

imag_t = input.imag()
# Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[6., 5., 4.],
#         [3., 2., 1.]])
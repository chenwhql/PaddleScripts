import paddle
import time
import yep

paddle.set_device("cpu")
input = paddle.rand([2, 2], 'float32')
input.stop_gradient=False

st = time.time()
yep.start("diagonal_dev.prof")
for i in range(500000):
    output = paddle.diagonal(input, 0, 0, 1 )
    # print("cur iter: ", i)
yep.stop()
et = time.time()

print(et - st)

# import paddle
# import time
# from paddle import _C_ops
# from paddle.fluid.framework import _test_eager_guard
# from paddle.fluid import core
# import yep
# #from core.eager.ops import final_state_trunc

# paddle.set_device("cpu")
# with _test_eager_guard():
#     input = paddle.rand([2,2],'float32')
#     #input.stop_gradient=False
#     st = time.time()
#     yep.start("diagonal.prof")
#     for i in range(1000000):
#         #output = paddle.trunc(input)
#         output = _C_ops.final_state_diagonal( input, 0, 0, 1 )
#     yep.stop()
#     et = time.time()
# print( et - st )
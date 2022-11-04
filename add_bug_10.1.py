import paddle
import os
from paddle import _C_ops

# from paddle.fluid.framework import _dygraph_tracer_, _in_eager_mode_
# print(_dygraph_tracer_)
# print(_in_eager_mode_)

# print(os.environ.get('FLAGS_enable_eager_mode', '1'))

t_pd_1 = paddle.randn([31280, 4], dtype='float32')
t_pd_2 = paddle.randn([31280, 4], dtype='float32')

# assert paddle.framework.in_dygraph_mode() is True

tt_pd = _C_ops.add(t_pd_1, t_pd_2)
# tt_pd = paddle.add(t_pd_1, t_pd_2)
print(tt_pd)

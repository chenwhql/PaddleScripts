import paddle
from paddle.utils.cpp_extension import load
import numpy as np
import os
import pytest
current_path = os.path.dirname(os.path.abspath(__file__))


paddle.set_device("cpu")
paddle.seed(33)
custom_ops = load(
    name="slice_op_jit",
    sources=[current_path + "/slice_op.cc"])


def test_add_op_jit():
    """
    test slice op jit
    Returns:

    """
    x = np.array([1, 2, 3, 4, 5, 6, 7])
    paddle_x = paddle.to_tensor(x).astype("float32")
    paddle_x.stop_gradient = False
    print(paddle_x)
    a = 1
    b = 5
    out = custom_ops.slice_test(paddle_x, a, b)
    print("out: ", out)
    print("numpy out: ", x[a:b])
    assert np.allclose(out.numpy(), x[a:b])
    print("run success")

test_add_op_jit()
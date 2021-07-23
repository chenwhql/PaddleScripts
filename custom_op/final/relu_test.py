import os
import numpy as np

import paddle
import paddle.static as static

from relu2 import load_custom_op, relu2

def test_relu2_dynamic(device, dtype):
    paddle.set_device(device)
    
    x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
    t = paddle.to_tensor(x)
    t.stop_gradient = False

    out = relu2(t)
    out.stop_gradient = False
    print(out.numpy())

    out.backward()

def test_relu2_static(device, dtype):
    paddle.enable_static()
    paddle.set_device(device)
    
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            out = relu2(x)
            static.append_backward(out)
            print(static.default_main_program())
            
            exe = static.Executor()
            exe.run(static.default_startup_program())
            
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out, = exe.run(
                static.default_main_program(),
                feed={'X': x},
                fetch_list=[out.name])
            print(out)

if __name__ == '__main__':
    load_custom_op("relu2_op.so")

    # dynamic graph mode
    test_relu2_dynamic("cpu", "float32")
    test_relu2_dynamic("cpu", "float64")
    test_relu2_dynamic("gpu", "float32")
    test_relu2_dynamic("gpu", "float64")

    # static graph mode
    test_relu2_static("cpu", "float32")
    test_relu2_static("cpu", "float64")
    test_relu2_static("gpu", "float32")
    test_relu2_static("gpu", "float64")
    
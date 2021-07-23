import os
import numpy as np

import paddle
import paddle.static as static
from paddle.utils.cpp_extension import load

custom_relu = load(
    name='custom_relu_jit_lib',
    sources=['relu_op.cc', 'relu_op.cu'])

def test_relu2_dynamic(device, dtype):
    paddle.set_device(device)
    
    x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
    t = paddle.to_tensor(x)
    t.stop_gradient = False

    out = custom_relu(t)
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
            out = custom_relu(x)
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
    # dynamic graph mode
    test_relu2_dynamic("gpu", "float32")
    test_relu2_dynamic("gpu", "float64")

    # static graph mode
    test_relu2_static("gpu", "float32")
    test_relu2_static("gpu", "float64")
    
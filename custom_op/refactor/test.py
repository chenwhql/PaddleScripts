import os
import numpy as np

import paddle
import paddle.static as static
import custom_relu_op_rf

def test_relu2_dynamic(device, dtype):
    paddle.set_device(device)
    
    x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
    t = paddle.to_tensor(x)
    t.stop_gradient = False

    out = custom_relu_op_rf.relu2(t)
    out.stop_gradient = False
    print(out.numpy())

    out.backward()

def test_relu2_static(device, dtype, use_custom=True):
    paddle.enable_static()
    paddle.set_device(device)
    
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            out = custom_relu_op_rf.relu2(x) if use_custom else paddle.nn.functional.relu(x)
            static.append_backward(out)
            print(static.default_main_program())
            
            places = static.cuda_places()
            print(places)
            exe = static.Executor()
            compiled_prog = static.CompiledProgram(
                static.default_main_program()).with_data_parallel(
                        loss_name=out.name, places=static.cuda_places())
            
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out, = exe.run(
                compiled_prog,
                feed={'X': x},
                fetch_list=[out.name])
            print(out)

if __name__ == '__main__':
    # dynamic graph mode
    # test_relu2_dynamic("gpu", "float32")
    # test_relu2_dynamic("gpu", "float64")

    # static graph mode
    test_relu2_static("gpu", "float32")
    # test_relu2_static("gpu", "float32", False)
    # test_relu2_static("gpu", "float64")
    
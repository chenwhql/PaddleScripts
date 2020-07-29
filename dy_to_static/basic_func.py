import numpy as np

import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import dygraph_to_static_func

def dyfunc_Linear(input):
    fc = fluid.dygraph.Linear(
        input_dim=10,
        output_dim=5,
        act='relu',
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.99)),
        bias_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.5)), )
    res = fc(input)
    return res

startup_program = fluid.Program()
main_program = fluid.Program()
with fluid.program_guard(main_program, startup_program):
    input = np.random.random((4, 3, 10)).astype('float32')
    data = fluid.layers.assign(input)
    static_out = dygraph_to_static_func(dyfunc_Linear)(data)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(startup_program)
static_res = exe.run(main_program, fetch_list=static_out)

print(static_res[0])
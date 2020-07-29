
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import data_layer_not_check

def static_func(x):
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x = fluid.layers.assign(x)
    x_shape = fluid.layers.shape(x)
    i = fluid.layers.fill_constant([1], 'int32', 0)

    def for_loop_condition_0(z, i):
        return i + 1 <= x_shape[0]

    def for_loop_body_0(z, i):
        z = z + x[i]
        i += 1
        return z, i

    z, i = fluid.layers.while_loop(for_loop_condition_0, for_loop_body_0, [z, i])
    return z

def execute():
    startup_program = fluid.Program()
    main_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        # prepare input
        np_x = np.array([1, 2, 3, 4, 5], dtype='int32')
        
        # add feed var
        x = data_layer_not_check(name='x', shape=list(np_x.shape), dtype=str(np_x.dtype))

        # build net
        result = static_func(x)

        print(main_program)

        # prepare exe
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        # run
        # exe.run(startup_program)
        out, = exe.run(main_program,
            feed={'x': np_x},
            fetch_list=[result])
        print(out[0])


if __name__== '__main__':
    execute()
    



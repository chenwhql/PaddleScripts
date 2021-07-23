from paddle import fluid
import paddle.fluid.transpiler.details.program_utils as pu

# net = lambda x : x * x

x = fluid.layers.create_parameter(shape=[1], dtype='float32', default_initializer=fluid.initializer.Constant(2))

y = fluid.layers.elementwise_mul(x, x)
grad1 = fluid.gradients(y, x)[0] # 2x = 4

# pu.program_to_code(fluid.default_main_program(), skip_op_callstack=True)

z = fluid.layers.elementwise_sub(x, grad1)
y2 = fluid.layers.elementwise_mul(z, z)
grad2 = fluid.gradients(y2, x)[0] # gradients( (x - 2x)^2) = 2x = 4

pu.program_to_code(fluid.default_main_program(), skip_op_callstack=True)
                                                                            
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

out_np = exe.run(fluid.default_main_program(), fetch_list=[grad1, grad2])
print(out_np)
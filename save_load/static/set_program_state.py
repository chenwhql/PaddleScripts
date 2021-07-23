import paddle
import paddle.fluid as fluid

paddle.enable_static()

x = fluid.data( name="x", shape=[10, 10], dtype='float32')
y = fluid.layers.fc( x, 10)
z = fluid.layers.fc( y, 10)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run( fluid.default_startup_program() )
prog = fluid.default_main_program()

fluid.save( prog, "./temp")
program_state = fluid.load_program_state( "./temp")

fluid.set_program_state( prog, program_state)
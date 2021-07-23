import paddle
import paddle.fluid as fluid

# paddle.enable_static()

prog = fluid.default_main_program()
fluid.save( prog, "./temp")
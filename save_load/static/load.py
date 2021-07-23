import paddle.fluid as fluid

prog = fluid.default_main_program()
fluid.save( prog, "./temp")

fluid.load( prog, "./temp")
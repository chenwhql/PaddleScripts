import paddle
import paddle.static as static

paddle.enable_static()

prog = static.default_main_program()
block_0 = prog.block(0)
print(block_0)
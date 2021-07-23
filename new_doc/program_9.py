import paddle
import paddle.static as static

paddle.enable_static()

prog = static.default_main_program()
num_blocks = prog.num_blocks
print(num_blocks)
import paddle
import paddle.static as static

paddle.enable_static()

prog = static.default_main_program()
gb_block = prog.global_block()
print(gb_block)
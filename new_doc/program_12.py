import paddle
import paddle.static as static

paddle.enable_static()

prog = static.default_main_program()
current_blk = prog.current_block()
print(current_blk)
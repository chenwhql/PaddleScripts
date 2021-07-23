import paddle
import paddle.static as static

paddle.enable_static()

prog = static.default_main_program()
print(prog.random_seed)
## 0
## the default random seed is 0

prog.global_seed(102)
prog1 = static.default_main_program()
print(prog1.random_seed)
## 102
## the random seed is 102
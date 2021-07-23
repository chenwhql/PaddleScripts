import paddle
import paddle.static as static
import paddle.nn.functional as F

paddle.enable_static()

prog = static.default_main_program()
random_seed = prog.random_seed
x_var = static.data(name="X", shape=[3,3], dtype="float32")
print(random_seed)
## 0
## the default random seed is 0

# Here we need to set random seed before we use fluid.layers.dropout
prog.random_seed = 1
z_var = F.dropout(x_var, 0.7)

print(prog.random_seed)
## 1
## the random seed is change to 1
import paddle.fluid as fluid

target_spos = fluid.layers.fill_constant(shape=[1], value=100, dtype="int32")
step_idx = fluid.layers.fill_constant(shape=[1], value=0, dtype="int64")
max_len = fluid.layers.fill_constant(shape=[1], value=3, dtype="int64")
cond = fluid.layers.less_than(x=step_idx, y=max_len)

while_op = fluid.layers.While(cond)

with while_op.block():
    decode_position = target_spos + step_idx

    fluid.layers.Print(step_idx)
    fluid.layers.Print(decode_position)

    fluid.layers.increment(x=step_idx, value=1, in_place=True)
    fluid.layers.less_than(x=step_idx, y=max_len, cond=cond)

exe = fluid.Executor(fluid.CUDAPlace(0))
exe.run(fluid.default_startup_program())
exe.run(fluid.default_main_program())
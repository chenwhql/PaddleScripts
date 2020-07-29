import paddle.fluid as fluid
import numpy as np
import paddle.fluid as fluid
import paddle

input_ = fluid.layers.data(name="input", shape=[-1, 1], lod_level=1,
append_batch_size=False, dtype="int64") # , stop_gradient=False)
label = fluid.layers.data(
name="label", shape=[-1, 1], append_batch_size=False, dtype="int64")

embed = fluid.embedding(
input=input_,
size=[100, 11],
dtype='float32')

embed_ = fluid.layers.sequence_pool(embed, pool_type="AVERAGE")

place = fluid.CPUPlace()
exe = fluid.Executor(place)

exe.run(fluid.default_startup_program())

input_value = fluid.create_lod_tensor(data=np.array(
[[1], [2], [3], [4]]), recursive_seq_lens=[[1, 3]], place=place)

embed_result = exe.run(fluid.default_main_program(), feed={
label: np.array([[0], [1]]), input_: input_value}, fetch_list=[embed_])
print(embed_result[0].shape)
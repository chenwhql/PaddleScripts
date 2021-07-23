import paddle.fluid as fluid
import paddle
import numpy as np

x = fluid.layers.data(name='x', shape=[1], dtype='float32', lod_level=1)
x = fluid.layers.Print(x, message="The content of input layer:")

y = fluid.layers.data(name='y', shape=[1], dtype='float32', lod_level=2)
out = fluid.layers.sequence_expand(x=x, y=y, ref_level=0)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
x_d = fluid.create_lod_tensor(np.array([[1.1], [2.2],[3.3],[4.4]]).astype('float32'), [[1,3]], place)
y_d = fluid.create_lod_tensor(np.array([[1.1],[1.1],[1.1],[1.1],[1.1],[1.1]]).astype('float32'), [[1,3], [1,2,1,2]], place)
results = exe.run(fluid.default_main_program(),
                  feed={'x':x_d, 'y': y_d },
                  fetch_list=[out],return_numpy=False)
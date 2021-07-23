import paddle.fluid as fluid
import numpy

place = fluid.CPUPlace()
exe = fluid.Executor(place)

data = fluid.layers.data(name='x', shape=[None, 5, 3], dtype='float32', lod_level=1)
y = fluid.layers.argmax(x=data, axis=-1)

x = numpy.random.random(size=(1, 5, 3)).astype('float32')
rlt = exe.run(fluid.default_main_program(),
              feed={"x": x},
              fetch_list=[data.name, y.name],
              return_numpy=False)

print(rlt)
print(rlt[0].lod())
print(rlt[1].lod())
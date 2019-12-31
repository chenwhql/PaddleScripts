import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import numpy
import os

place = fluid.CUDAPlace(0) # fluid.CPUPlace()
exe = fluid.Executor(place)

data = fluid.layers.data(name='X', shape=[1], dtype='float32')
hidden = fluid.layers.fc(input=data, size=10)
loss = fluid.layers.mean(hidden)
fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

"""
print("startup program:")
print(fluid.default_startup_program())
print("main program:")
print(fluid.default_main_program())
"""

test_prog = fluid.default_main_program().clone(for_test=True)
print(test_prog)

fluid.default_startup_program().random_seed=1
exe.run(fluid.default_startup_program())
compiled_prog = compiler.CompiledProgram(
                 fluid.default_main_program())

x = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = exe.run(compiled_prog,
                     feed={"X": x},
                     fetch_list=[loss.name])

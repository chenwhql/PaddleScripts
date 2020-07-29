import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import numpy
import os

place = fluid.CUDAPlace(0) # fluid.CPUPlace()
exe = fluid.Executor(place)

data = fluid.layers.data(name='X', shape=[1], dtype='float32')
hidden = fluid.layers.fc(input=data, size=10)
fluid.layers.Print(hidden)
loss = fluid.layers.mean(hidden)
fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

exe.run(fluid.default_startup_program())

main_program = fluid.default_main_program()
print(main_program)

x = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = exe.run(main_program,
                     feed={"X": x},
                     fetch_list=[loss.name],
                     use_prune=True)

# need print in executor
# print(main_program)

print("loss: {}".format(loss_data[0]))

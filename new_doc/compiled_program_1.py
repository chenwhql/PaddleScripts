import numpy
import paddle
import paddle.static as static

paddle.enable_static()

place = paddle.CUDAPlace(0) # paddle.CPUPlace()
exe = static.Executor(place)

data = static.data(name='X', shape=[None, 1], dtype='float32')
hidden = static.nn.fc(input=data, size=10)
loss = paddle.mean(hidden)
paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

exe.run(static.default_startup_program())
compiled_prog = static.CompiledProgram(
    static.default_main_program())

x = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=[loss.name])
import numpy
import os
import paddle
import paddle.static as static

paddle.enable_static()

use_cuda = True
place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
exe = static.Executor(place)

# NOTE: If you use CPU to run the program, you need
# to specify the CPU_NUM, otherwise, paddle will use
# all the number of the logic core as the CPU_NUM,
# in that case, the batch size of the input should be
# greater than CPU_NUM, if not, the process will be
# failed by an exception.
if not use_cuda:
    os.environ['CPU_NUM'] = str(2)
    places = static.cpu_places()
else:
    places = static.cuda_places()

data = static.data(name='X', shape=[None, 1], dtype='float32')
hidden = static.nn.fc(input=data, size=10)
loss = paddle.mean(hidden)
paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

exe.run(static.default_startup_program())

build_strategy = static.BuildStrategy()
build_strategy.gradient_scale_strategy = \
          static.BuildStrategy.GradientScaleStrategy.Customized
compiled_prog = static.CompiledProgram(
          static.default_main_program()).with_data_parallel(
                  loss_name=loss.name, build_strategy=build_strategy,
                  places=places)

dev_count =  len(places)
x = numpy.random.random(size=(10, 1)).astype('float32')
loss_grad = numpy.ones((dev_count)).astype("float32") * 0.01
loss_grad_name = loss.name+"@GRAD"
loss_data = exe.run(compiled_prog,
                      feed={"X": x, loss_grad_name : loss_grad},
                      fetch_list=[loss.name, loss_grad_name])
import numpy
import os
import paddle
import paddle.static as static

paddle.enable_static()

use_cuda = True
place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
parallel_places = [paddle.CUDAPlace(0), paddle.CUDAPlace(1)] if use_cuda else [paddle.CPUPlace()] * 2

# NOTE: If you use CPU to run the program, you need
# to specify the CPU_NUM, otherwise, paddle will use
# all the number of the logic core as the CPU_NUM,
# in that case, the batch size of the input should be
# greater than CPU_NUM, if not, the process will be
# failed by an exception.
if not use_cuda:
    os.environ['CPU_NUM'] = str(2)

exe = static.Executor(place)

data = static.data(name='X', shape=[None, 1], dtype='float32')
hidden = static.nn.fc(input=data, size=10)
loss = paddle.mean(hidden)

test_program = static.default_main_program().clone(for_test=True)
paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

exe.run(static.default_startup_program())
compiled_train_prog = static.CompiledProgram(
    static.default_main_program()).with_data_parallel(
            loss_name=loss.name, places=parallel_places)
# NOTE: if not set share_vars_from=compiled_train_prog,
# the parameters used in test process are different with 
# the parameters used by train process
compiled_test_prog = static.CompiledProgram(
    test_program).with_data_parallel(
            share_vars_from=compiled_train_prog,
            places=parallel_places)

train_data = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = exe.run(compiled_train_prog,
                  feed={"X": train_data},
                  fetch_list=[loss.name])
test_data = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = exe.run(compiled_test_prog,
                  feed={"X": test_data},
                  fetch_list=[loss.name])
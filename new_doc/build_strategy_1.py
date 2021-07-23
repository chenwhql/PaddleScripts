import os
import paddle
import paddle.static as static

paddle.enable_static()

os.environ['CPU_NUM'] = str(2)
places = static.cpu_places()

data = static.data(name="x", shape=[None, 1], dtype="float32")
hidden = static.nn.fc(input=data, size=10)
loss = paddle.mean(hidden)
paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

build_strategy = static.BuildStrategy()
build_strategy.enable_inplace = True
build_strategy.memory_optimize = True
build_strategy.reduce_strategy = static.BuildStrategy.ReduceStrategy.Reduce
program = static.CompiledProgram(static.default_main_program())
program = program.with_data_parallel(loss_name=loss.name,
                                      build_strategy=build_strategy,
                                      places=places)
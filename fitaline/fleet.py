# Include libraries.
import os
import sys

import numpy

import paddle
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import StrategyFactory
from paddle.fluid.reader import keep_data_loader_order

import local

batch_size = 24

# keep_data_loader_order(False)

# Define train function.
def train():
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    label = fluid.layers.data(name='y', shape=[1], dtype='float32')

    py_reader = fluid.io.PyReader(
        feed_list=[x, label], capacity=64, use_double_buffer=False, iterable=False)

    avg_cost = local.net(x, label)

    exe = fluid.Executor(fluid.CPUPlace())
    role = role_maker.PaddleCloudRoleMaker()

    fleet.init(role)

    strategy = StrategyFactory.create_async_strategy()
    optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(avg_cost)

    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()

    if fleet.is_worker():
        fleet.init_worker()

        py_reader.decorate_batch_generator(local.fake_reader())
        exe.run(fleet.startup_program)

        prog = fluid.compiler.CompiledProgram(fleet.main_program).with_data_parallel(
                loss_name=avg_cost.name,
                build_strategy=strategy.get_build_strategy(),
                exec_strategy=strategy.get_execute_strategy())

        PASS_NUM = 10
        for pass_id in range(PASS_NUM):
            batch = 0
            py_reader.start()

            try:
                while True:
                    avg_loss_value, = exe.run(prog, fetch_list=[avg_cost])
                    print("pass %d, batch %d, total avg cost = %f" % (pass_id, batch, avg_loss_value))
                    batch += 1
            except fluid.core.EOFException:
                py_reader.reset()

        fleet.stop_worker()


# Run train and infer.
if __name__ == '__main__':
    train()


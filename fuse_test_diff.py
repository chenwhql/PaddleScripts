import paddle.fluid as fluid
import numpy as np
import random

batch_size = 32

feed_dict = {
    'image': np.random.random([batch_size, 784]).astype('float32'),
    'label': np.random.random_integers(
        low=0, high=9, size=[batch_size, 1]).astype('int64')
}

def simple_fc_net():
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    prediction = fluid.layers.fc(img, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss

def build_program_and_scope():
    startup_program = fluid.Program()
    main_program = fluid.Program()
    startup_program.random_seed = 1
    main_program.random_seed = 1

    scope = fluid.Scope()
    with fluid.program_guard(main_program, startup_program):
        with fluid.unique_name.guard():
            loss = simple_fc_net()
            adam = fluid.optimizer.SGD(learning_rate=1e-3)
            adam.minimize(loss)

            with fluid.scope_guard(scope):
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
    return main_program, scope, exe, loss

# Program
prog1, scope1, exe, loss1 = build_program_and_scope()

# CompiledProgram
prog2, scope2, _, loss = build_program_and_scope()
build_strategy = fluid.BuildStrategy()
# if close this strategy, no diff
build_strategy.fuse_all_optimizer_ops = True
build_strategy.debug_graphviz_path = "./"
exec_strategy = fluid.ExecutionStrategy()
exec_strategy.num_threads = 1
compiled_prog = fluid.CompiledProgram(prog2).with_data_parallel(
    loss_name=loss.name,
    build_strategy=build_strategy,
    exec_strategy=exec_strategy,
    places=fluid.CPUPlace())

for i in range(4):
    with fluid.scope_guard(scope1):
        fetch_val1, = exe.run(prog1,
                              feed=feed_dict,
                              fetch_list=['fc_0.b_0'])

    with fluid.scope_guard(scope2):
        fetch_val2, = exe.run(compiled_prog,
                              feed=feed_dict,
                              fetch_list=['fc_0.b_0'])

        if not np.array_equal(fetch_val1, fetch_val2):
            print("Iter: %d" % i)
            for i in range(len(fetch_val1)):
                if(fetch_val1[i] != fetch_val2[i]):
                    print("index: %d, val1: %.12f, val2: %.12f" % (i, fetch_val1[i], fetch_val2[i]))

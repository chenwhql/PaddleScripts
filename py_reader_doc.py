import paddle
import paddle.fluid as fluid
import paddle.dataset.mnist as mnist

def network(reader):
    img, label = fluid.layers.read_file(reader)
    # 用户自定义网络，此处以softmax回归为例
    predict = fluid.layers.fc(input=img, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=predict, label=label)
    return fluid.layers.mean(loss)

# 新建 train_main_prog 和 train_startup_prog
train_main_prog = fluid.Program()
train_startup_prog = fluid.Program()
with fluid.program_guard(train_main_prog, train_startup_prog):
    # 使用 fluid.unique_name.guard() 实现与test program的参数共享
    with fluid.unique_name.guard():
        train_reader = fluid.layers.py_reader(capacity=64,
                                              shapes=[(-1, 1, 28, 28), (-1, 1)],
                                              dtypes=['float32', 'int64'],
                                              name='train_reader')
        train_reader.decorate_paddle_reader(
        paddle.reader.shuffle(paddle.batch(mnist.train(),
                              batch_size=5),
                              buf_size=500))
        train_loss = network(train_reader) # 一些网络定义
        adam = fluid.optimizer.Adam(learning_rate=0.01)
        adam.minimize(train_loss)

# Create test_main_prog and test_startup_prog
test_main_prog = fluid.Program()
test_startup_prog = fluid.Program()
with fluid.program_guard(test_main_prog, test_startup_prog):
    # 使用 fluid.unique_name.guard() 实现与train program的参数共享
    with fluid.unique_name.guard():
        test_reader = fluid.layers.py_reader(capacity=32,
                                             shapes=[(-1, 1, 28, 28), (-1, 1)],
                                             dtypes=['float32', 'int64'],
                                             name='test_reader')
        test_reader.decorate_paddle_reader(paddle.batch(mnist.test(), 512))
        test_loss = network(test_reader)

fluid.Executor(fluid.CUDAPlace(0)).run(train_startup_prog)
fluid.Executor(fluid.CUDAPlace(0)).run(test_startup_prog)

train_exe = fluid.ParallelExecutor(use_cuda=True,
    loss_name=train_loss.name, main_program=train_main_prog)
test_exe = fluid.ParallelExecutor(use_cuda=True,
    loss_name=test_loss.name, main_program=test_main_prog)
for epoch_id in range(10):
    train_reader.start()
    try:
        while True:
            train_exe.run(fetch_list=[train_loss.name])
    except fluid.core.EOFException:
        train_reader.reset()

test_reader.start()
try:
    while True:
        test_exe.run(fetch_list=[test_loss.name])
except fluid.core.EOFException:
    test_reader.reset()
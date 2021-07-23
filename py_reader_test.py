import paddle
import paddle.fluid as fluid
import paddle.dataset.mnist as mnist

def network(image, label):
    # user defined network, here a softmax regession example
    predict = fluid.layers.fc(input=image, size=10, act='softmax')
    return fluid.layers.cross_entropy(input=predict, label=label)

reader = fluid.layers.py_reader(capacity=64,
                                shapes=[(-1, 1, 28, 28), (-1, 1)],
                                dtypes=['float32', 'int64'])
reader.decorate_paddle_reader(
    paddle.reader.shuffle(paddle.batch(mnist.train(), batch_size=5),
                          buf_size=10))

img, label = fluid.layers.read_file(reader)
loss = network(img, label)

fluid.Executor(fluid.CPUPlace()).run(fluid.default_startup_program())
exe = fluid.Executor(fluid.CPUPlace()) # fluid.ParallelExecutor(use_cuda=True)
for epoch_id in range(1):
    reader.start()
    try:
        while True:
            loss_var = exe.run(fetch_list=[loss.name])
            print(loss_var[0])
    except fluid.core.EOFException:
        reader.reset()

fluid.io.save_inference_model(dirname='./model',
                              feeded_var_names=[img.name, label.name],
                              target_vars=[loss],
                              executor=fluid.Executor(fluid.CPUPlace()))
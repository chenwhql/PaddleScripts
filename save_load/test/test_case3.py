import numpy as np
import contextlib

import paddle
import paddle.fluid as fluid

'''
Global Configs
'''
USE_CUDA = True
EPOCH_NUM = 2
BATCH_SIZE = 64
MODEL_PATH = "mnist.declarative.to.imperative"

'''
Part 1. Model
'''
def convolutional_neural_network(img):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction

def static_train_net(img, label):
    prediction = convolutional_neural_network(img)

    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)

    optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    return prediction, avg_loss

'''
Part 2. Assist Functions
'''
@contextlib.contextmanager
def new_program_scope():
    prog = fluid.Program()
    startup_prog = fluid.Program()
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        with fluid.program_guard(prog, startup_prog):
            yield

def reader_decorator(reader):
    def __reader__():
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(1, 28, 28)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label

    return __reader__

'''
Part 3. Train & Save
'''
startup_program = fluid.default_startup_program()
main_program = fluid.default_main_program()

img = fluid.data(
    name='img', shape=[None, 1, 28, 28], dtype='float32')
label = fluid.data(name='label', shape=[None, 1], dtype='int64')
prediction, avg_loss = static_train_net(img, label)

place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_program)

# create train data loader
train_reader = paddle.batch(
    paddle.dataset.mnist.train(),
    batch_size=BATCH_SIZE,
    drop_last=True)
train_loader = fluid.io.DataLoader.from_generator(feed_list=[img, label], capacity=5)
train_loader.set_sample_list_generator(train_reader, places=place)

for epoch in range(EPOCH_NUM):
    for batch_id, data in enumerate(train_loader()):
        results = exe.run(main_program,
                feed=data,
                fetch_list=[avg_loss])

        if batch_id % 200 == 0:
            print("Loss at step {}: {:}".format(
                batch_id, results[0]))

fluid.io.save_inference_model(
    MODEL_PATH, ["img"], [prediction], exe)

'''
Part 4. Load & Inference
'''
image = np.random.random((1, 1, 28, 28)).astype('float32')
# load model in static mode & inference
with new_program_scope():
    [infer_program, feed, fetch] = fluid.io.load_inference_model(MODEL_PATH, exe)
    static_pred = exe.run(infer_program,
        feed={feed[0]: image},
        fetch_list=fetch)
# load model in dygraph mode & inference
with fluid.dygraph.guard(place):
    loaded_mnist = fluid.dygraph.jit.load(model_path=MODEL_PATH)
    loaded_mnist.eval()
    image_var = fluid.dygraph.to_variable(image)
    dygraph_pred = loaded_mnist(image_var)
# compare
print("static prediction: {}".format(static_pred[0]))
print("dygraph prediction: {}".format(dygraph_pred.numpy()))
np.testing.assert_array_equal(dygraph_pred.numpy(), static_pred[0])

'''
Part 5. Load & Fine-tune
'''
# train in static mode
for batch_id, data in enumerate(train_loader()):
    static_loss = exe.run(main_program,
            feed=data,
            fetch_list=[avg_loss])

    if batch_id % 200 == 0:
        print("Loss at step {}: {:}".format(
            batch_id, static_loss[0]))
# train in dygraph mode
with fluid.dygraph.guard(place):
    loaded_mnist = fluid.dygraph.jit.load(model_path=MODEL_PATH)
    loaded_mnist.train()
    sgd = fluid.optimizer.SGD(learning_rate=0.001,
                              parameter_list=loaded_mnist.parameters())

    backward_strategy = fluid.dygraph.BackwardStrategy()
    backward_strategy.sort_sum_gradient = True

    # create train data loader
    train_reader = paddle.batch(
        reader_decorator(paddle.dataset.mnist.train()),
        batch_size=BATCH_SIZE,
        drop_last=True)
    dy_train_loader = fluid.io.DataLoader.from_generator(capacity=5)
    dy_train_loader.set_sample_list_generator(train_reader, places=place)

    for batch_id, data in enumerate(dy_train_loader()):
        img, label = data
        label.stop_gradient = True
        
        cost = loaded_mnist(img)
        loss = fluid.layers.cross_entropy(cost, label)
        dygraph_loss = fluid.layers.mean(loss)

        dygraph_loss.backward(backward_strategy)
        sgd.minimize(dygraph_loss)
        loaded_mnist.clear_gradients()

        if batch_id % 200 == 0:
            print("Loss at step {}: {:}".format(
                batch_id, dygraph_loss.numpy()))
# compare
print("static loss: {}".format(static_loss[0]))
print("dygraph loss: {}".format(dygraph_loss.numpy()))
np.testing.assert_allclose(static_loss[0], dygraph_loss.numpy(), rtol=1e-5, atol=0)
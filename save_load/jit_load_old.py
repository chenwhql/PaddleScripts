import numpy as np
import paddle.fluid as fluid

BATCH_SIZE = 32
BATCH_NUM = 20
SAVE_DIRNAME = "fc.inference.model"

def random_batch_reader():
    def _get_random_images_and_labels(image_shape, label_shape):
        image = np.random.random(size=image_shape).astype('float32')
        label = np.random.random(size=label_shape).astype('int64')
        return image, label

    def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = _get_random_images_and_labels(
                [BATCH_SIZE, 784], [BATCH_SIZE, 1])
            yield batch_image, batch_label

    return __reader__

def train_and_save_static_model(place):
    img = fluid.data(name='img', shape=[None, 784], dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')

    pred = fluid.layers.fc(input=img, size=10, act='softmax')

    loss = fluid.layers.cross_entropy(input=pred, label=label)
    avg_loss = fluid.layers.mean(loss)

    optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    loader = fluid.io.DataLoader.from_generator(
        feed_list=[img, label], capacity=5, iterable=True)
    loader.set_batch_generator(random_batch_reader(), places=place)

    for data in loader():
        exe.run(
            fluid.default_main_program(),
            feed=data, 
            fetch_list=[avg_loss])

    # save model by fluid.io.save_inference_model
    fluid.io.save_inference_model(
        SAVE_DIRNAME, ["img"], [pred], exe)


# Step 1. train and save inference model in static graph mode
place = fluid.CPUPlace()
# train_and_save_static_model(place)

# Step 2. load inference model in dygraph and fine-tune
with fluid.dygraph.guard(place):
    fc = fluid.dygraph.jit.load(SAVE_DIRNAME)

    sgd = fluid.optimizer.SGD(learning_rate=0.001,
                            parameter_list=fc.parameters())

    train_loader = fluid.io.DataLoader.from_generator(capacity=5)
    train_loader.set_batch_generator(
        random_batch_reader(), places=place)

    for data in train_loader():
        img = data[0]
        label = data[1]
        label.stop_gradient = True

        cost = fc(inputs=img)

        loss = fluid.layers.cross_entropy(cost, label)
        avg_loss = fluid.layers.mean(loss)

        avg_loss.backward()
        sgd.minimize(avg_loss)
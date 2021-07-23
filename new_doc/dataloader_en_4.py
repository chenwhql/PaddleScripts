import paddle
import paddle.static as static

paddle.enable_static()

image = static.data(name='image', shape=[None, 784], dtype='float32')
label = static.data(name='label', shape=[None, 1], dtype='int64')

dataset = paddle.distributed.QueueDataset()
dataset.init(
    batch_size=32,
    pipe_command='cat',
    use_var=[image, label])
dataset.set_filelist(['a.txt', 'b.txt', 'c.txt'])

loader = paddle.io.DataLoader.from_dataset(dataset, static.cpu_places())
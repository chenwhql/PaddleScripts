import paddle
import paddle.fluid as fluid

def num_generator():
    def __reader__():
        for i in range(100):
            yield i
    return __reader__

reader = paddle.batch(
    paddle.shuffle(num_generator(), buf_size=10),
    batch_size=20,
    drop_last=True)

for data in reader():
    print(data)
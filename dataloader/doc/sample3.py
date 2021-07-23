'''
Example of `drop_last` using in static graph multi-cards mode
'''
import paddle
import paddle.static as static
import numpy as np
import os

# We use 2 CPU cores to run inference network
os.environ['CPU_NUM'] = '2'

paddle.enable_static()

# The data source has only 3 batches, which can not be
# divided evenly to each CPU core
def batch_generator():
    for i in range(3):
        yield np.array([i+1]).astype('float32'),

x = static.data(name='x', shape=[None], dtype='float32')
y = x * x

def run_inference(drop_last):
    loader = paddle.io.DataLoader.from_generator(feed_list=[x],
            capacity=8, drop_last=drop_last)
    loader.set_batch_generator(batch_generator, static.cpu_places())

    exe = static.Executor(paddle.CPUPlace())
    prog = static.CompiledProgram(static.default_main_program())
    prog = prog.with_data_parallel()

    result = []
    for data in loader():
        each_ret, = exe.run(prog, feed=data, fetch_list=[y])
        result.extend(each_ret)
    return result

# Set drop_last to True, so that the last batch whose
# number is less than CPU core number would be discarded.
print(run_inference(drop_last=True)) # [1.0, 4.0]

# Set drop_last to False, so that the last batch whose
# number is less than CPU core number can be tested.
print(run_inference(drop_last=False)) # [1.0, 4.0, 9.0]
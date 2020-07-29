import paddle.fluid as fluid
from paddle.fluid.io import multiprocess_reader
import numpy as np

sample_files = ['sample_file_1', 'sample_file_2']

def fake_input_files():
    with open(sample_files[0], 'w') as f:
        np.savez(f, a=np.array([1, 2]), b=np.array([3, 4]), c=np.array([5, 6]), d=np.array([7, 8]))
    with open(sample_files[1], 'w') as f:
        np.savez(f, a=np.array([9, 10]), b=np.array([11, 12]), c=np.array([13, 14]))


def generate_reader(file_name):
    # load data file
    def _impl():
        data = np.load(file_name)
        for item in sorted(data.files):
            yield data[item],
    return _impl

if __name__ == '__main__':
    # generate sample input files
    fake_input_files()
    
    with fluid.program_guard(fluid.Program(), fluid.Program()):
        place = fluid.CPUPlace()
        # the 1st 2 is batch size
        image = fluid.data(name='image', dtype='int64', shape=[2, 1, 2]) 
        fluid.layers.Print(image)
        # print detailed tensor info of image variable
    
        reader = fluid.io.PyReader(feed_list=[image], capacity=2)
    
        decorated_reader = multiprocess_reader(
            [generate_reader(sample_files[0]), generate_reader(sample_files[1])], False)
    
        reader.decorate_sample_generator(decorated_reader, batch_size=2, places=[place])
    
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
    
        for data in reader():
            res = exe.run(feed=data, fetch_list=[image])
            print(res[0])
            # print below content in this case
            # [[[1 2]], [[3 4]]]
            # [[[5 6]], [[7 8]]]
            # [[[9 10]], [[11 12]]]
            # [13,14] will be dropped
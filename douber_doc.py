import paddle.fluid as fluid
reader = fluid.layers.py_reader(capacity=64,
                                shapes=[(-1, 1, 28, 28), (-1, 1)],
                                dtypes=['float32', 'int64'],
                                use_double_buffer=False)
reader = fluid.layers.double_buffer(reader)
image, label = fluid.layers.read_file(reader)
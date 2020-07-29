import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, to_variable, TracedLayer
import numpy as np

class ExampleLayer(fluid.dygraph.Layer):
    def __init__(self):
        super(ExampleLayer, self).__init__()
        self._fc = Linear(3, 10)

    def forward(self, input):
        return self._fc(input)

save_dirname = './saved_infer_model'
in_np = np.random.random([2, 3]).astype('float32')

with fluid.dygraph.guard():
    layer = ExampleLayer()
    in_var = to_variable(in_np)
    out_dygraph, static_layer = TracedLayer.trace(layer, inputs=[in_var])
    print(static_layer.program)

    for i in range(10):
        in_var = to_variable(in_np)
        # print(in_var.name)
        out_var = static_layer([in_var])
        # print(in_var.name)
        print(out_var[0].name)

    static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])

place = fluid.CPUPlace()
exe = fluid.Executor(place)
program, feed_vars, fetch_vars = fluid.io.load_inference_model(save_dirname,
                                    exe)

fetch, = exe.run(program, feed={feed_vars[0]: in_np}, fetch_list=fetch_vars)
print(fetch.shape) # (2, 10)
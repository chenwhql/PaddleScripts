import paddle
import paddle.fluid as fluid
import numpy as np

paddle.enable_static()

# Build the model
main_prog = fluid.Program()
startup_prog = fluid.Program()
with fluid.program_guard(main_prog, startup_prog):
    data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
    w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32')
    b = fluid.layers.create_parameter(shape=[200], dtype='float32')
    hidden_w = fluid.layers.matmul(x=data, y=w)
    hidden_b = fluid.layers.elementwise_add(hidden_w, b)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_prog)

# Save the inference model
path = "./infer_model"
fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'],
              target_vars=[hidden_b], executor=exe, main_program=main_prog)

# Demo one. Not need to set the distributed look up table, because the
# training doesn't use a distributed look up table.
[inference_program, feed_target_names, fetch_targets] = (
    fluid.io.load_inference_model(dirname=path, executor=exe))
tensor_img = np.array(np.random.random((1, 64, 784)), dtype=np.float32)
results = exe.run(inference_program,
              feed={feed_target_names[0]: tensor_img},
              fetch_list=fetch_targets)

# Demo two. If the training uses a distributed look up table, the pserver
# endpoints list should be supported when loading the inference model.
# The below is just an example.
endpoints = ["127.0.0.1:2023","127.0.0.1:2024"]
[dist_inference_program, dist_feed_target_names, dist_fetch_targets] = (
    fluid.io.load_inference_model(dirname=path,
                                  executor=exe,
                                  pserver_endpoints=endpoints))

# In this example, the inference program was saved in the file
# "./infer_model/__model__" and parameters were saved in
# separate files under the directory "./infer_model".
# By the inference program, feed_target_names and
# fetch_targets, we can use an executor to run the inference
# program for getting the inference result.
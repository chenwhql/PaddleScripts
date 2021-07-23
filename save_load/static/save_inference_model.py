import paddle
import paddle.fluid as fluid

paddle.enable_static()

path = "./infer_model"

# User defined network, here a softmax regession example
image = fluid.data(name='img', shape=[None, 28, 28], dtype='float32')
label = fluid.data(name='label', shape=[None, 1], dtype='int64')
feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
predict = fluid.layers.fc(input=image, size=10, act='softmax')

loss = fluid.layers.cross_entropy(input=predict, label=label)
avg_loss = fluid.layers.mean(loss)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

# Feed data and train process

# Save inference model. Note we don't save label and loss in this example
fluid.io.save_inference_model(dirname=path,
                              feeded_var_names=['img'],
                              target_vars=[predict],
                              executor=exe)

# In this example, the save_inference_mode inference will prune the default
# main program according to the network's input node (img) and output node(predict).
# The pruned inference program is going to be saved in the "./infer_model/__model__"
# and parameters are going to be saved in separate files under folder
# "./infer_model".
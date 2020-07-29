import numpy as np
import paddle.fluid as fluid

# enable dygraph mode
fluid.enable_dygraph() 

# use SaveLoadconfig when loading model
model_path = "simplenet.example.model"
configs = fluid.dygraph.jit.SaveLoadConfig()
configs.model_filename = "__simplenet__"
infer_net = fluid.dygraph.jit.load(model_path, configs=configs)
# inference
x = fluid.dygraph.to_variable(np.random.random((4, 8)).astype('float32'))
pred = infer_net(x)
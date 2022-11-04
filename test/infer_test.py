import os
import readline
import cv2 as cv
import numpy as np
import paddle.fluid as fluid
import paddle
from paddle.fluid.executor import global_scope

input_size=np.array([672,672]).reshape((1,2)).astype('int32')
resize_size=672



model_path="./model/"
exe = fluid.Executor(fluid.CPUPlace())
paddle.enable_static()
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe, 'model-1.pdmodel', 'model-1.pdiparams')
scope = paddle.static.global_scope()
print("----") 

for var in inference_program.list_vars():
    var.persistable = True

img = np.ones((1,3,672,672)).astype("float32")
img=img.reshape((1,3,672,672))
#np.save("../data_for_lite/img.bin",img)

# fetch_targets = fetch_targets.append("conv1_3.conv2d.output.1.tmp_0")
# print(fetch_targets)
results = exe.run(inference_program, feed={feed_target_names[0]: img,feed_target_names[1]: input_size}, fetch_list=fetch_targets, return_numpy=False)
print(results[0])
# print(results[1])

print("----")
# kid_scopes = scope._kids()
# print("kid scope num: ", len(kid_scopes))
var_node = scope.find_var("conv1_3.conv2d.output.1.tmp_0")
assert var_node is not None, \
    "Cannot find "  + " in scope."
print("----") 
print("tmp:",np.array(var_node.get_tensor()))
       
    
        
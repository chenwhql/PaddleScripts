import paddle
import paddle.fluid as fluid
import cv2
import numpy as np

paddle.enable_static()

img = cv2.imread("test1.jpg")
# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

save_path = './'
[infer_program,
  feeded_var_names,
  target_var] = fluid.io.load_inference_model(dirname=save_path,
                                              executor=exe,
                                              params_filename="__params__")
print(infer_program)

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.resize(img, (112, 112))
img = np.array([img]).astype(np.float32)
# img = np.array([1, 1, 112, 112]).astype(np.float32)
img = img / 255.0

face = img[np.newaxis, :]
height = np.array([169.0]).astype(np.float32)[np.newaxis, :]

result = exe.run(program=infer_program,
                  feed={feeded_var_names[0]: face, feeded_var_names[1]: height},
                  fetch_list=target_var)

print(result)
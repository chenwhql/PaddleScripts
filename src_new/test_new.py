
import numpy as np
import paddle.fluid as fluid
import os
import sys
import cv2
import time
import shapely
from shapely.geometry import Polygon


import paddle
from paddle.utils.cpp_extension import load

custom_ops = load(
    name="custom_jit_ops",
    sources=["rbox_iou_op.cc", "rbox_iou_op.cu"])

paddle.set_device('gpu')
paddle.disable_static()

rbox1 = [[772.0, 575.5, 90.75791931152344, 26.0, -0.3468348975810076]]
rbox1 = np.array(rbox1)

rbox2 = [[772.0, 575.5, 90.75791931152344, 26.0, 0.0]]
rbox2 = np.array(rbox2)

# use_rand_data = True
use_rand_data = False

if use_rand_data:
    rbox1 = np.random.rand(13000, 5)
    rbox2 = np.random.rand(7, 5)

    # x1 y1 w h [0, 0.5]
    rbox1[:, 0:4] = rbox1[:, 0:4] * 0.45 + 0.001
    rbox2[:, 0:4] = rbox2[:, 0:4] * 0.45 + 0.001

    rbox1[:, 4] = rbox1[:, 4] - 0.5
    rbox2[:, 4] = rbox2[:, 4] - 0.5


print('rbox1', rbox1.shape, 'rbox2', rbox2.shape)

print('rbox1', rbox1)
print('rbox2', rbox2)

print("before to tensor")

pd_rbox1 = paddle.to_tensor(rbox1, dtype="float32")
pd_rbox2 = paddle.to_tensor(rbox2, dtype="float32")

print("end to tensor")

# print(pd_rbox1)
# print(pd_rbox2)

res = custom_ops.rbox_iou(pd_rbox1, pd_rbox2)
start_time = time.time()
print(pd_rbox1)
print(pd_rbox2)
print('paddle cpu cuda\n\n')
res = custom_ops.rbox_iou(pd_rbox1, pd_rbox2)
print('paddle time:', time.time() - start_time)
print(res.cpu().shape)
print(res.cpu())




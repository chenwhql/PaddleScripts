'''
Author: your name
Date: 2021-07-07 10:14:46
LastEditTime: 2021-07-08 12:02:22
LastEditors: Please set LastEditors
Description: In User Settings Ed
FilePath: /scripts/op2func/sign.py
'''

import paddle
import numpy as np

np_x = np.array([-1., 0., -0., 1.2, 1.5], dtype='float64')
x = paddle.to_tensor(np_x)
z = paddle.sign(x)
print(z)
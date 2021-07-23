import time

import paddle

paddle.set_device('cpu') # gpu

a = paddle.ones([512, 3, 124, 124], dtype='int32')  
b = 2.0

s_time = time.time()
for i in range(1000):
    a = a + b
    a.numpy()
e_time = time.time()

avg_time = (e_time - s_time) / 1000

print(avg_time)
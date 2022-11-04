import paddle
import paddle.nn as nn
import numpy as np

paddle.set_device("gpu")

x = np.array([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]).astype('float32')
x = paddle.to_tensor(x)

sync_batch_norm = nn.SyncBatchNorm(2)
sync_batch_norm.eval()

hidden1 = sync_batch_norm(x)
out = paddle.mean(hidden1)
print("out: ", out)

out.backward()
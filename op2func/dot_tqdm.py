import paddle
import numpy as np
from tqdm import tqdm

paddle.set_device("cpu")
x_data = np.random.uniform(0.1, 1, [10]).astype(np.float32)
y_data = np.random.uniform(1, 3, [10]).astype(np.float32)

x = paddle.to_tensor(x_data)
y = paddle.to_tensor(y_data)

for i in tqdm(range(1000000)):
    z = paddle.dot(x, y)

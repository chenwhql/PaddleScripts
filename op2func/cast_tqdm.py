import paddle
import numpy as np
from tqdm import tqdm

paddle.set_device("cpu")
x = paddle.to_tensor([2, 3, 4], 'float64')

for i in tqdm(range(1000000)):
    y = paddle.cast(x, 'uint8')

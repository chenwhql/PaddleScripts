import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from tqdm import tqdm
import time

class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, obs):
        # h1 = F.relu(self.fc1(obs))
        # out = F.relu(self.fc2(h1))
        h1 = self.fc1(obs)
        out = self.fc2(h1)
        return out

paddle.set_device('cpu')

mymodel = LinearNet()
mymodel.eval()

t = time.time()

data_np = np.random.rand(1,1)
data_in = paddle.to_tensor(data_np.astype(np.float32))
for _ in tqdm(range(100000)):
    # out = mymodel(data_in)
    # out = paddle.reshape(data_in, [-1])
    out = data_in + data_in

cost = time.time() - t
print(cost)
# 3.278186321258545
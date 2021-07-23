import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
torch.set_num_threads(1)

class LinearNet(nn.Module):
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

mymodel = LinearNet()

t = time.time()

data_np = np.random.rand(1,1)
data_in = torch.tensor(data_np, dtype=torch.float)
for _ in tqdm(range(100000)):
    # out = mymodel(data_in)
    # out = torch.reshape(data_in, [-1])
    out = data_in + data_in

cost = time.time() - t
print(cost)
# 0.5656709671020508
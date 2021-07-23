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
        h1 = F.relu(self.fc1(obs))
        out = F.relu(self.fc2(h1))
        return out

mymodel = LinearNet()
data_np = np.random.rand(32, 4)
data_in = torch.tensor(data_np, dtype=torch.float)

t = time.time()
for _ in range(10):
# for _ in tqdm(range(100000)):
    out = mymodel(data_in)
cost = time.time() - t
print(cost)
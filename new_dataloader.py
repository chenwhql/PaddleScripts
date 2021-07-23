import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, BatchSampler, DataLoader

BATCH_NUM = 20
BATCH_SIZE = 16
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

USE_GPU = False # whether use GPU to run model

# define a random dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

paddle.enable_static()

dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)

loader = DataLoader(dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=True,
                    num_workers=2)

for e in range(EPOCH_NUM):
    for i, (image, label) in enumerate(loader()):
        print(type(image))
        print(image.__array__())
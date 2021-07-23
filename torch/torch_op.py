import torch
import numpy as np

a = torch.full([2, 2, 2], 2, dtype=torch.float32)
b = 3.0
c = a ** b
print(c)
print(c.dtype)

a = 2.5
b = torch.full((2, 2), 2.0, dtype=torch.float32)
c = a // b
print(c)

# a = 2.5
# b = torch.full((2, 2), 2.0, dtype=torch.float32)
# c = a % b
# print(c)

a = 3
b = torch.full((2, 2), 7)
c = a / b
print(c)
print(c.dtype)

bn = np.full((2, 2), 7)
cn = a / bn
print(cn)
print(cn.dtype)

print(np.array_equal(c, cn))
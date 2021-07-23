import torch

t = torch.ones((2, 4), dtype=torch.float64)
out = torch.real(t)
print(out)
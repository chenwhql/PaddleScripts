import torch

a = torch.randn([2, 3, 4])
print(a)

b = a[0, 0, 0, None, None]
print(b)


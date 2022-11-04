import torch

a = torch.Tensor()
print(a)

b = torch.Tensor(3)
print(b)
print(b.device)
print(b[0])
print(b[0].dim())
print(b[0].device)

b = torch.tensor([1, 2, 3], device=torch.device('cuda:0'))
print(b)
print(b.device)
print(b[0])
print(b[0].dim())
print(b[0].device)

print(b[0].numel())
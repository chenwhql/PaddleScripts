# import torch

# a = torch.tensor(2.)
# print(a.requires_grad)
# b = torch.tensor(3., requires_grad=True)

# torch.Tensor.add_(a, b)
# print(a)
# print(a.requires_grad)

#####

import torch

a = torch.tensor(2., requires_grad=False)
print(a.requires_grad)
b = torch.tensor(3., requires_grad=True)

torch.Tensor.add_(a, b)
print(a)
print(a.requires_grad)
a.retain_grad()

a.sum().backward()
print(a.grad)

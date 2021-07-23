import torch 

a = torch.ones([2, 3, 4])
# b = [[True, False, True], [False, True, False]]
# b = [True, False]
# b = [0, 1]
print(a[[0, 1]])

# print(a[b])
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 12, 3)
    def forward(self, x, y):
        x = self.conv(x)
        y = self.conv(y)
        var = y.flatten()
        x[0, :, 0, 0] *= var
        print(x)
        loss = torch.mean(x)
        var.register_hook(lambda grad: print('var grad', grad.sum()))
        return loss, var

model = Model()

x = torch.ones([1, 12, 3, 3]).float()
y = torch.ones([1, 12, 3, 3]).float()
loss, var = model(x, y)

loss.backward()

print(var.grad.sum())
import torch
import numpy as np

def print_hook(grad):
    print("grad:", grad)

x = np.random.random([2, 3]).astype(np.complex64) + 1j * np.random.random(
    [2, 3]).astype(np.complex64)

y = np.random.random([2, 3]).astype(np.complex64) + 1j * np.random.random(
    [2, 3]).astype(np.complex64)

x = torch.tensor(x).requires_grad_()
x.retain_grad()
y = torch.tensor(y).requires_grad_()
y.retain_grad()

print("x:", x)
print("y:", y)

out = torch.mul(x, y)
print("out:", out)

# loss = torch.mean(out)

real = torch.real(out)
loss = torch.mean(real)

# imag = torch.imag(out)
# loss = torch.mean(imag)

x.register_hook(print_hook)
out.register_hook(print_hook)
loss.register_hook(print_hook)

loss.backward()
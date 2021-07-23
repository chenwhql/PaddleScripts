import torch

def print_hook(grad):
    print("grad:", grad)

t = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]).requires_grad_()
t.retain_grad()

out = torch.imag(t)
# out = torch.real(t)
print("out:", out)

t.register_hook(print_hook)
out.register_hook(print_hook)

out.backward(torch.ones(3))

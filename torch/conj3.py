import torch

def print_hook(grad):
    print("grad:", grad)

t = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]).requires_grad_()
t.retain_grad()

out = torch.conj(t)
print("out:", out)

t.register_hook(print_hook)
out.register_hook(print_hook)

out.backward(torch.tensor([1+1j, 1+1j, 1+1j]))
# out.backward(torch.ones(3, dtype=torch.complex64))
# out.backward(torch.ones(3))
# out.backward()


import torch

def print_hook(grad):
    print("grad:", grad)

t = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]).requires_grad_()
t.retain_grad()

out = torch.conj(t)
print("out:", out)

loss = torch.mean(out)

# real = torch.real(out)
# loss = torch.mean(real)

# imag = torch.imag(out)
# loss = torch.mean(imag)

t.register_hook(print_hook)
out.register_hook(print_hook)
loss.register_hook(print_hook)

loss.backward()

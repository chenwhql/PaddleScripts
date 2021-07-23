import torch

def print_fn(grad):
    print(grad)

def double_fn(grad):
    grad = grad * 2
    print(grad)
    return grad

x = torch.ones([1], dtype=torch.float32).requires_grad_()
x.register_hook(double_fn)

y = x * x

dx = torch.autograd.grad(
    outputs=[y],
    inputs=[x],
    create_graph=False,
    retain_graph=True)[0]

z = y + dx
# z.backward()

print("x.grad:", x.grad)
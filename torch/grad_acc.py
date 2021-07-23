import torch

a = torch.Tensor([0, 1, 1, 2]).requires_grad_()
b = torch.Tensor([0, 0, 1, 2]).requires_grad_()
x = a + b
x.retain_grad()

a.register_hook(lambda x: 2*x)

y = torch.Tensor([4, 5, 6, 7]).requires_grad_()
z = torch.Tensor([1, 2, 3, 4]).requires_grad_()

x.register_hook(lambda x: 2*x)

o1 = x + y 
o2 = x + z
o1.retain_grad()
o2.retain_grad()

print(o1)
print(o2)

o = o1.matmul(o2)
o.retain_grad()

print(o)

o.backward()

# grad print
print('a.grad: ', a.grad)
print('b.grad: ', b.grad)
print('x.grad: ', x.grad)
print('y.grad: ', y.grad)
print('z.grad: ', z.grad)
print('o1.grad: ', o1.grad)
print('o2.grad: ', o2.grad)
print('o.grad: ', o.grad)

# grad_fn print
print('x.grad_fn: ', x.grad_fn)
print('y.grad_fn: ', y.grad_fn)
print('z.grad_fn: ', z.grad_fn)
print('o1.grad_fn: ', o1.grad_fn)
print('o2.grad_fn: ', o2.grad_fn)
print('o.grad_fn: ', o.grad_fn)


# grad_fn next_functions
print('x.grad_fn.next_functions: ', x.grad_fn.next_functions)
print('o1.grad_fn.next_functions: ', o1.grad_fn.next_functions)
print('o2.grad_fn.next_functions: ', o2.grad_fn.next_functions)
print('o.grad_fn.next_functions: ', o.grad_fn.next_functions)

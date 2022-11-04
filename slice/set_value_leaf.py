import paddle

m0 = paddle.zeros((2, 2))
m0.stop_gradient = False
m = m0 * 2.0
m.stop_gradient = False
print(m)
print(m.is_leaf) # True
a = paddle.to_tensor(2., stop_gradient=False)
b = paddle.to_tensor(3., stop_gradient=False)

m[0, 1] = a * b

loss = m.sum()
paddle.autograd.grad(loss, a)
paddle.autograd.backward(loss, a)
print("a.grad: ", a.grad)
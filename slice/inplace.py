import paddle

a = paddle.to_tensor(2.)
b = paddle.to_tensor(3., stop_gradient=False)

paddle.tensor.add_(a, b)
print(a)

####

# import paddle

# # a = paddle.to_tensor(2., stop_gradient=False)
# a = paddle.to_tensor(2., stop_gradient=True)
# b = paddle.to_tensor(3., stop_gradient=False)

# paddle.tensor.add_(a, b)
# print(a)

# a.sum().backward()
# print(a.grad)
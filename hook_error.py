from __future__ import print_function

import paddle

# hook function return None
def print_hook_fn(grad):
    print(grad)
    # print("inf", paddle.has_inf(grad))
    print("inf", paddle.fluid.layers.has_inf(grad))

# hook function return Tensor
def double_hook_fn(grad):
    grad = grad * 2
    return grad

x = paddle.to_tensor([0., 1., 2., 3.], stop_gradient=False)
y = paddle.to_tensor([4., 5., 6., 7.], stop_gradient=False)
z = paddle.to_tensor([1., 2., 3., 4.])

# one Tensor can register multiple hooks
h = x.register_hook(print_hook_fn)
x.register_hook(double_hook_fn)

w = x + y
# register hook by lambda function
w.register_hook(lambda grad: grad * 2)

o = z.matmul(w)
o.backward()
# print_hook_fn print content in backward
# Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
#        [2., 4., 6., 8.])

print("w.grad:", w.grad) # w.grad: [1. 2. 3. 4.] - no changed
print("x.grad:", x.grad) # x.grad: [ 4.  8. 12. 16.] - deal with by two *2 hook
print("y.grad:", y.grad) # y.grad: [2. 4. 6. 8.] - - deal with by one *2 hook
# remove hook
h.remove()
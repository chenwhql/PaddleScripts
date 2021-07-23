import torch 
import paddle
import numpy as np

def func(t,value):
    a = t * t
    a.retain_grad()
    print(a.grad_fn)

    # a[0] = value
    # print(a.grad_fn)
    # print(value.grad_fn)

    b = a[0]
    b.retain_grad()
    print(b.grad_fn)

    b = value
    print(b.grad_fn)
    print(a.grad_fn)
    
    return a.sum()


array=np.array([1,2,3,4], dtype='float32')
value=np.array([5.],dtype='float32')

tt=torch.tensor(array,requires_grad=True)
tvalue=torch.tensor(value,requires_grad=True)

l1=func(tt,tvalue)
l1.backward()

print('torch:array',tt.grad)
print('torch:value',tvalue.grad)
# print('torch:a',l1[1].retain_grad())


# pt=paddle.to_tensor(array,stop_gradient=False)
# pvalue=paddle.to_tensor(value,stop_gradient=False)

# l2=func(pt,pvalue)
# l2.backward()
# print('paddle:array',pt.grad)
# print('paddle:value',pvalue.grad)
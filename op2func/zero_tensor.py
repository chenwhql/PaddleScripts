import paddle

a = paddle.to_tensor([1, 2, 3])
print(a[0].shape) # expected: []
print(a[0].numel()) # expected: 1
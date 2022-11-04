import paddle

x = paddle.to_tensor([1, 4, 5, 2])
out = paddle.diff(x)
print(out)
# out:
# [3, 1, -3]

y = paddle.to_tensor([7, 9])
out = paddle.diff(x, append=y)
print(out)
# out:
# [3, 1, -3, 5, 2]

z = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
print(z.shape)
out = paddle.diff(z, axis=0)
print(out)
# out:
# [[3, 3, 3]]

temp1 = paddle.slice(z, [1], [0], [2])
temp2 = paddle.slice(z, [1], [1], [3])
print(temp1)
print(temp2)
print(temp2 - temp1)

# tmp1 = paddle._C_ops.final_state_slice(z, [1], [0], [2], [1], [])
# tmp2 = paddle._C_ops.final_state_slice(z, [1], [1], [3], [1], [])
# print(tmp1)
# print(tmp2)
# print(tmp2 - tmp1)

out = paddle.diff(z, axis=1)
print(out)
# out:
# [[1, 1], [1, 1]]
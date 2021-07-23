import paddle

x1 = paddle.to_tensor([[0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0]])
out_z1 = paddle.nonzero(x1)
print(out_z1)
print(len(out_z1))
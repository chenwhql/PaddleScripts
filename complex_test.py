import paddle

x = paddle.randn((10,))+paddle.to_tensor(1j)*paddle.randn((10,)) 

print(x)

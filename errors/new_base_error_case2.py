import paddle

# 执行设备配置
paddle.set_device('cpu')

# 网络配置
linear = paddle.nn.Linear(1, 10)

# 数据准备
x = paddle.randn(shape=(10, 2), dtype='float32')

# 网络执行
res = linear(x)
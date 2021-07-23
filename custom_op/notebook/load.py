import paddle
from paddle.utils.cpp_extension import load

# 即时编译
custom_ops = load(
    name="custom_jit_ops",
    sources=["relu_cpu.cc"],
    verbose=True)

# 使用API
paddle.set_device('cpu')
x = paddle.randn([4, 10], dtype='float32')
out = custom_ops.custom_relu(x)
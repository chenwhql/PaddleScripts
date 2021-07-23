import paddle
from paddle.utils.cpp_extension import load

# jit compile custom op
custom_ops = load(
    name="custom_jit_ops",
    sources=["relu_cuda.cu"],
    verbose=True)


x = paddle.randn([4, 10], dtype='float32')
relu_out = custom_ops.custom_relu(x)
print(relu_out)
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, setup

setup(
    name='custom_relu_lib',
    ext_modules=[
        CUDAExtension(
            name='custom_relu_op',
            sources=['relu_op.cc', "relu_op.cu"])
    ])
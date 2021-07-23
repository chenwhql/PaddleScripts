from paddle.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, setup

setup(
    name='relu2_op_shared',
    ext_modules=[
        CUDAExtension(
            name='librelu2_op',
            sources=['relu_op.cc'])
    ])
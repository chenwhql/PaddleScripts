from paddle.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, setup

# setup(
#     name='custom_relu_lib_rf',
#     ext_modules=[
#         CUDAExtension(
#             name='custom_relu_op_rf',
#             sources=['relu_op.cc', "relu_op.cu"])
#     ])

setup(
    name='custom_relu_lib_rf',
    ext_modules=[
        CppExtension(
            name='custom_relu_op_rf',
            sources=['relu_op.cc'])
    ])
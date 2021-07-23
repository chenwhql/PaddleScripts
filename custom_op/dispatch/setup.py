from paddle.utils.cpp_extension import CppExtension, setup

setup(
    name='dispatch_op',
    ext_modules=[
        CppExtension(
            name='dispatch_op',
            sources=['diapatch_test_op.cc'])
    ])
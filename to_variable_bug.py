import paddle
tensor=paddle.randn((3,4))
dlpack=tensor.value().get_tensor()._to_dlpack()
tensor_from_dlpack = paddle.fluid.core.from_dlpack(dlpack)
paddle.fluid.dygraph.to_variable(tensor_from_dlpack)
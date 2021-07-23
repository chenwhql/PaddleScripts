import paddle

t = paddle.randn([2,3])
print(t)
print(type(t))
print(t.__dict__)

t.__class__ = paddle.fluid.framework.ParamBase
print(t)
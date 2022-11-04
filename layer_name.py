import paddle

paddle.set_default_dtype("float64")

class UCIHousing(paddle.nn.Layer):
    def __init__(self):
        super(UCIHousing, self).__init__()

        # weight_attr = paddle.ParamAttr(
        #         # name="weight",
        #         initializer=paddle.nn.initializer.Constant(value=0.5))
        # bias_attr = paddle.ParamAttr(
        #         # name="bias",
        #         initializer=paddle.nn.initializer.Constant(value=1.0))

        # weight_attr = paddle.ParamAttr(
        #         name=paddle.utils.unique_name.generate("weight"),
        #         initializer=paddle.nn.initializer.Constant(value=0.5))
        # bias_attr = paddle.ParamAttr(
        #         name=paddle.utils.unique_name.generate("bias"),
        #         initializer=paddle.nn.initializer.Constant(value=1.0))

        weight_attr = paddle.ParamAttr(
                name="weight",
                initializer=paddle.nn.initializer.Constant(value=0.5))
        bias_attr = paddle.ParamAttr(
                name="bias",
                initializer=paddle.nn.initializer.Constant(value=1.0))

        self.fc = paddle.nn.Linear(10, 5, weight_attr=weight_attr, bias_attr=bias_attr, name = "x_fc")

    def forward(self, input):
        pred = self.fc(input)
        return pred

with paddle.utils.unique_name.guard():
    model_1st = UCIHousing()
    print(model_1st.parameters())

# 这里报错:（同结构的不同模型训练，参数名称固定，不同的对象为什么不能重复定义？）
with paddle.utils.unique_name.guard():
    model_2nd = UCIHousing()
    print(model_2nd.parameters())

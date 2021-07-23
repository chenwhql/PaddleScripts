# import paddle

# fc1 = paddle.nn.Linear(10, 3)
# fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(fc1, fc2)
# for prefix, layer in model.named_sublayers():
#     print(prefix, layer)

import paddle

class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self._linear = paddle.nn.Linear(1, 1)
        self._dropout = paddle.nn.Dropout(p=0.5)

    def forward(self, input):
        temp = self._linear(input)
        temp = self._dropout(temp)
        return temp

mylayer = MyLayer()
print(mylayer.sublayers())
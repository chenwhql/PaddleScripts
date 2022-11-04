import paddle
from copy import deepcopy

model = paddle.nn.Layer()
print(id(model))

model_c = deepcopy(model)

print(id(model_c))
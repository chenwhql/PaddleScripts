import paddle
import numpy as np
paddle.disable_static()
value = np.arange(26).reshape(2, 13).astype("float32")
a = paddle.to_tensor(value)
linear = paddle.nn.Linear(13, 5, dtype="float32")
# This can be any optimizer supported by dygraph.
adam = paddle.optimizer.Adam(learning_rate = 0.01, 
                            parameters = linear.parameters())
out = linear(a)
out.backward()
adam.step()
adam.clear_grad()
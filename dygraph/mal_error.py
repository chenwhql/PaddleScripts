import paddle.fluid as fluid

import numpy as np

fluid.enable_dygraph()

t = np.sqrt(2.0 * np.pi)

print( t) 
print( type(t) )
x = fluid.layers.ones( (2, 2), dtype="float32")

y = t * x 
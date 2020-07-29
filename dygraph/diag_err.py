import paddle.fluid as fluid
import numpy as np

def test_diag():
    """
    test diag

    Returns:
        None
    """
    with fluid.dygraph.guard():
        diagonal = np.arange(3, 6, dtype='int64')
        data = fluid.layers.diag(diagonal)
        expect = [[3, 0, 0], [0, 4, 0], [0, 0, 5]]
        print(data)

test_diag()
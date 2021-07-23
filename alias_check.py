import paddle.fluid as fluid

with fluid.dygraph.guard():
    emb = fluid.dygraph.Embedding([10, 10])
    # with self.assertRaises(AttributeError):
    emb.__aliases__
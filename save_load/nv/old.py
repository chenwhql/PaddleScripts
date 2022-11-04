class cuDNNMultiHeadAttention(paddle.nn.Layer):
    def _init__(self, ...):
        ...

    def forward(...):
        ...

    @classmethod
    def to_legacy(...):
        ...

layer = cuDNNMultiHeadAttention()

# hidden this line
layer = cuDNNMultiHeadAttention.to_legacy(layer, ...)

paddle.jit.save(layer, ...)

class cuDNNMultiHeadAttention(paddle.nn.Layer):
    def _init__(self, ...):
        ...

        paddle.jit.register_save_pre_hook(
            cuDNNMultiHeadAttention.to_legacy)

    def forward(...):
        ...

    @classmethod
    def to_legacy(...):
        ...

layer = cuDNNMultiHeadAttention()
paddle.jit.save(layer, ...)

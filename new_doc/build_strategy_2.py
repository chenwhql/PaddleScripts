import paddle
import paddle.static as static

paddle.enable_static()

build_strategy = static.BuildStrategy()
build_strategy.reduce_strategy = static.BuildStrategy.ReduceStrategy.Reduce
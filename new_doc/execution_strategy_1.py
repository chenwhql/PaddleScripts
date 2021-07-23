import paddle
import paddle.static as static
import paddle.nn.functional as F

paddle.enable_static()

x = static.data(name='x', shape=[None, 13], dtype='float32')
y = static.data(name='y', shape=[None, 1], dtype='float32')
y_predict = static.nn.fc(input=x, size=1, act=None)

cost = F.square_error_cost(input=y_predict, label=y)
avg_loss = paddle.mean(cost)

sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_loss)

exec_strategy = static.ExecutionStrategy()
exec_strategy.num_threads = 4

train_exe = static.ParallelExecutor(use_cuda=False,
                                    loss_name=avg_loss.name,
                                    exec_strategy=exec_strategy)
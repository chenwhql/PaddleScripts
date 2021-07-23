import paddle
import paddle.static as static

paddle.enable_static()

program = static.default_main_program()
data = static.data(name='x', shape=[None, 13], dtype='float32')
hidden = static.nn.fc(input=data, size=10)
loss = paddle.mean(hidden)
paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

for param in program.all_parameters():
    print(param)
import paddle.fluid as fluid

program = fluid.default_main_program()
data = fluid.data(name='x', shape=[None, 13], dtype='float32')
hidden = fluid.layers.fc(input=data, size=10)
loss = fluid.layers.mean(hidden)
fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

for param in program.all_parameters():
    print(param)
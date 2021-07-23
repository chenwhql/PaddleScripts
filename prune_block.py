import paddle.fluid as fluid
import numpy as np
import paddle.fluid.transpiler.details.program_utils as pu

x_in = np.random.random(size=(10, 4)).astype('float32')
label_in = np.random.randint(1, size=(10, 1)).astype('int64')

x = fluid.layers.data(name="x",shape=[4],dtype='float32')
label = fluid.layers.data('label', shape=[1], dtype='int64')

#y = x + x

prediction = fluid.layers.fc(input=x, size=1, act=None)

def loss1(opt, pred, label):
    x = fluid.layers.data(name="x",shape=[4],dtype='float32')
    loss = fluid.layers.cross_entropy(input=pred, label=label)
    avg_loss = fluid.layers.mean(loss, name='mean_cross_entropy_loss')
    opt.minimize(avg_loss)
    return avg_loss

def loss2(opt, pred, label):
    loss = fluid.layers.softmax_with_cross_entropy(logits=pred, label=label)
    avg_loss = fluid.layers.mean(loss, name='mean_softmax_loss')
    opt.minimize(avg_loss)
    return avg_loss

sgd = fluid.optimizer.SGD(learning_rate=0.1)
two = fluid.layers.fill_constant([1], 'int32', 2)
pred = (two == 0)

avg_loss = fluid.layers.case([(
                pred, lambda: loss1(sgd, prediction, label))],
                                   lambda: loss2(sgd, prediction, label))

#sgd.minimize(avg_loss)
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)

exe.run(fluid.default_startup_program())
origin = fluid.default_main_program()

print("backward: %d" % fluid.default_main_program()._appending_grad_times)

pu.program_to_code(fluid.default_main_program(), skip_op_callstack=True)

print("original:")
for block in origin.blocks:
    print([op.type for op in block.ops])
#    print([var for var in block.vars])

test_prog = origin.clone(for_test=True)
pu.program_to_code(test_prog, skip_op_callstack=True)

print("after prune:")
for block in test_prog.blocks:
    print([op.type for op in block.ops])
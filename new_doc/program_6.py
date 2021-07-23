import six
import paddle
import paddle.static as static
import paddle.utils as utils
import paddle.nn.functional as F

paddle.enable_static()

def print_prog(prog):
    for name, value in sorted(six.iteritems(prog.block(0).vars)):
        print(value)
    for op in prog.block(0).ops:
        print("op type is {}".format(op.type))
        print("op inputs are {}".format(op.input_arg_names))
        print("op outputs are {}".format(op.output_arg_names))
        for key, value in sorted(six.iteritems(op.all_attrs())):
            if key not in ['op_callstack', 'op_role_var']:
                print(" [ attrs: {}:   {} ]".format(key, value))

def network():
    img = static.data(name='image', shape=[None, 784])
    hidden = static.nn.fc(input=img, size=200, act='relu')
    hidden = F.dropout(hidden, p=0.5)
    loss = F.cross_entropy(
        input=static.nn.fc(hidden, size=10, act='softmax'),
        label=static.data(name='label', shape=[1], dtype='int64'))
    avg_loss = paddle.mean(loss)
    return avg_loss

train_program_2 = static.Program()
startup_program_2 = static.Program()
test_program_2 = static.Program()
with static.program_guard(train_program_2, startup_program_2):
    with utils.unique_name.guard():
        avg_loss = network()
        sgd = paddle.optimizer.SGD(learning_rate=1e-3)
        sgd.minimize(avg_loss)
# the test startup program is not used.
with static.program_guard(test_program_2, startup_program_2):
    with utils.unique_name.guard():
        avg_loss = network()
print_prog(test_program_2)
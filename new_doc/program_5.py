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

train_program = static.Program()
startup_program = static.Program()

# startup_program is used to do some parameter init work,
# and main program is used to hold the network
with static.program_guard(train_program, startup_program):
    with utils.unique_name.guard():
        img = static.data(name='image', shape=[None, 784])
        hidden = static.nn.fc(input=img, size=200, act='relu')
        hidden = F.dropout(hidden, p=0.5)
        loss = F.cross_entropy(
            input=static.nn.fc(hidden, size=10, act='softmax'),
            label=static.data(name='label', shape=[1], dtype='int64'))
        avg_loss = paddle.mean(loss)
        test_program = train_program.clone(for_test=True)
print_prog(test_program)

# Due to parameter sharing usage for train and test, so we need to use startup program of train
# instead of using test startup program, while nothing is in test's startup program

# In Paddle Fluid we will share weights by using the same Variable name. In train and test program
# all parameters will have the same name and this can make train and test program sharing parameters,
# that's why we need to use startup program of train. And for startup program of test, it has nothing,
# since it is a new program.

with static.program_guard(train_program, startup_program):
    with utils.unique_name.guard():
        sgd = paddle.optimizer.SGD(learning_rate=1e-3)
        sgd.minimize(avg_loss)
import paddle
import paddle.static as static

paddle.enable_static()

main_program = static.Program()
startup_program = static.Program()
with static.program_guard(main_program=main_program, startup_program=startup_program):
    x = static.data(name="x", shape=[-1, 784], dtype='float32')
    y = static.data(name="y", shape=[-1, 1], dtype='int32')
    z = static.nn.fc(name="fc", input=x, size=10, act="relu")

print("main program is: {}".format(main_program))
print("start up program is: {}".format(startup_program))
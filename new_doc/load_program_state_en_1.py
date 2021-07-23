import paddle
import paddle.static as static

paddle.enable_static()

x = static.data(name="x", shape=[10, 10], dtype='float32')
y = static.nn.fc(x, 10)
z = static.nn.fc(y, 10)

place = paddle.CPUPlace()
exe = static.Executor(place)
exe.run(static.default_startup_program())
prog = static.default_main_program()

static.save(prog, "./temp")
program_state = static.load_program_state("./temp")
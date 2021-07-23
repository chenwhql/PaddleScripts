import paddle
import paddle.static as static

paddle.enable_static()

startup_prog = static.Program()
main_prog = static.Program()
with static.program_guard(startup_prog, main_prog):
    x = static.data(name='X', shape=[1000, 784], dtype='float32')

    y = static.data(name='Y', shape=[784, 100], dtype='float32')

    z = paddle.matmul(x=x, y=y)

    binary_str = static.default_main_program().desc.serialize_to_string()
    prog_restored = static.default_main_program().parse_from_string(binary_str)

    print(static.default_main_program())
    print(prog_restored)
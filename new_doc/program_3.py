import paddle
import paddle.static as static

paddle.enable_static()

prog = static.default_main_program()
x = static.data(name="X", shape=[2,3], dtype="float32")
pred = static.nn.fc(x, size=3)
prog_string = prog.to_string(throw_on_error=True, with_details=False)
prog_string_with_details = prog.to_string(throw_on_error=False, with_details=True)
print("program string without detail: {}".format(prog_string))
print("program string with detail: {}".format(prog_string_with_details))
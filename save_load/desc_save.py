import paddle
import paddle.static as static
paddle.enable_static()
def build_program():
    main_program = paddle.static.Program()
    startuo_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program, startuo_program):
            x = paddle.static.data(name='x', shape=[3, 2, 1])
            out = paddle.static.nn.fc(x=x, size=1, num_flatten_dims=2)
            # for i in range(10):
            #     out = out + out
        return main_program

program = build_program()

print(program)
a = program.desc.serialize_to_string()
with open('a.pb', 'wb') as f:
    f.write(a)
load_main = paddle.load('a.pb')
print(load_main)
# program.current_block().ops[0]._set_attr('use_mkldnn', True)
# program.current_block().ops[0]._set_attr('scale_x', 2.0)
program.desc.block(0).op(0)._set_attr('use_mkldnn', True)
program.desc.block(0).op(0)._set_attr('scale_x', 2.0)
# program.desc.block(0).var("x").set_stop_gradient(True)
# program.current_block().vars["x"].desc.set_stop_gradient(False)
program.current_block().vars["x"].desc.set_persistable(True)

# program.desc.block(0).op(0).flush()
# with open('b.pb', 'wb') as f:
#     f.write(program.desc.serialize_to_string())
# # load_main = paddle.load('b.pb')

# origin_program_bytes = static.io.load_from_file('b.pb')
# load_main = static.io.deserialize_program(origin_program_bytes)
# print(load_main)

seri_prog = static.io._serialize_program(program)
recover_prog = static.io.deserialize_program(seri_prog)

print(program)
# print(program.to_string(False))
print(recover_prog)
# print(recover_prog.to_string(False))

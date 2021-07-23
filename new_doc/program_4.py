import six

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
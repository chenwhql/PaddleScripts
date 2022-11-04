import paddle

all_op_names = paddle.framework.core.get_all_op_names()
for op_name in all_op_names:
    print(op_name)

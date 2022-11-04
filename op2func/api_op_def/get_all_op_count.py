import paddle

all_op_names = paddle.framework.core.get_all_op_names()
print("前反向Op总个数：", len(all_op_names))

all_fwd_op_list = []
all_bwd_op_list = []
for op_name in all_op_names:
    if op_name.endswith("_grad") or op_name.endswith("_grad2"):
        all_bwd_op_list.append(op_name)
    else:
        all_fwd_op_list.append(op_name)

print("前向Op个数：", len(all_fwd_op_list))
print("反向Op个数：", len(all_bwd_op_list))

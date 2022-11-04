import paddle

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

print("全量Op个数：", len(all_fwd_op_list))
print("反向Op个数：", len(all_bwd_op_list))

attr_back_list = {'op_role', 'op_role_var', 'op_namescope', 'op_callstack', 'op_device', 'with_quant_attr', "use_mkldnn", "use_cudnn"}
extra_op_dict = dict()

for op_name in all_fwd_op_list:
    proto = paddle.fluid.framework.OpProtoHolder.instance().get_op_proto(type)
    for attr in proto.attrs:
        if attr.extra - True and attr.name not in attr_back_list:
            if not extra_op_dict.has_key(op_name):
                extra_op_dict[op_name] = []
            extra_op_dict[op_name].append(extra)



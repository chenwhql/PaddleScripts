import paddle

from paddle.fluid.framework import OpProtoHolder

def get_op_def(op_name):
    proto = OpProtoHolder.instance().get_op_proto(op_name)

    input_names = []
    for in_proto in proto.inputs:
        input_names.append(in_proto.name)

    output_names = []
    for out_proto in proto.outputs:
        output_names.append(out_proto.name)

    attr_names = []
    for attr_proto in proto.attrs:
        attr_names.append(attr_proto.name)

    def_str = "{}\t{}\t{}\t{}".format(
        op_name,
        ", ".join(s for s in input_names),
        ", ".join(s for s in output_names),
        ", ".join(s for s in attr_names))
    print(def_str)
    
invalid_set = set()
invalid_set.add("push_sparse_v2")
invalid_set.add("push_gpups_sparse")
invalid_set.add("push_box_sparse")
invalid_set.add("push_sparse")
invalid_set.add("push_box_extended_sparse")

all_op_names = paddle.framework.core.get_all_op_names()
for op_name in all_op_names:
    if op_name.endswith("_grad") or op_name.endswith("_grad2"):
        continue
    if op_name in invalid_set:
        continue
    get_op_def(op_name)

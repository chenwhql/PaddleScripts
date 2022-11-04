import paddle
import pandas as pd
import numpy as np

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

api_related_ops = pd.read_excel('api-related-op.xlsx', sheet_name='Sheet1', engine='openpyxl')
api_related_op_list = []
for row in api_related_ops.itertuples(index=False, name="Op"):
    op_name = row[0]
    api_related_op_list.append(op_name)

for op_name in api_related_op_list:
    get_op_def(op_name)

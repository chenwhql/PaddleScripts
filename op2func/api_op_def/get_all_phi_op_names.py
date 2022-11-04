import paddle

all_op_names = paddle.framework.core.get_all_op_names()
all_op_names_set = set(all_op_names)
print("Op numbers with phi kernel：", len(all_op_names_set))
# print(all_op_names_set)

all_phi_op_names = paddle.framework.core.get_all_op_names("phi")
all_phi_op_names_set = set(all_phi_op_names)
print("Op numbers with phi kernel：", len(all_phi_op_names_set))
# print(all_phi_op_names_set)

all_fluid_op_names = paddle.framework.core.get_all_op_names("fluid")
all_fluid_op_names_set = set(all_fluid_op_names)
print("Op numbers with fluid kernel：", len(all_fluid_op_names_set))
# print(all_fluid_op_names_set)

support_two_type_kernel_op_names_set = all_phi_op_names_set & all_fluid_op_names_set
print("Op numbers with fluid and phi kernel：", len(support_two_type_kernel_op_names_set))
# print(support_two_type_kernel_op_names_set)

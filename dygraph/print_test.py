
import gast
import astor

node1 = gast.parse("x = fluid.layers.Print(x, summarize=-1, print_phase='forward')")
# node2 = gast.parse("x = print(x)")
node3 = gast.parse("assert isinstance(x, Variable)")
# node4 = gast.parse("b = 1 or print('3')")
# node5 = gast.parse("True and print(x)")
print(astor.dump_tree(node5))

# b = 1 or a = print("2")
# 1 and 2 and 3 and x = print(4)

True and print(x)
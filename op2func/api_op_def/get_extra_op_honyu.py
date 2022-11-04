import sys

import paddle

#print("11")

from paddle.fluid import framework


file = open( "list.txt" )

set1 = dict()

back_list = { 'op_role', 'op_role_var', 'op_namescope', 'op_callstack', 'op_device', 'with_quant_attr', "use_mkldnn", "use_cudnn" }

for line in file:
    #print( line.strip() )

    type=line.strip()
    proto = framework.OpProtoHolder.instance().get_op_proto(type)


    for attr in proto.attrs:
        #print( attr.name )
        #print( attr.extra )
        if attr.extra == True and attr.name not in back_list:
            #print( attr.name )
            if attr.name in set1:
                set1[ attr.name ].append( type )
            else:
                set1[attr.name] = [  type ]


dict_2 = dict()
for k, v in set1.items():
    print(k, v )

    for e in v:
        if e in dict_2:
            dict_2[e].append( k )
        else:
            dict_2[e] = [ k ]


print("=================================")
for k, v in dict_2.items():
    print( k, "\t", v)

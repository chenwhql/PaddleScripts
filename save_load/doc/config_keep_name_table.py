import paddle
            
paddle.disable_static()

linear = paddle.nn.Linear(5, 1)

state_dict = linear.state_dict()
paddle.save(state_dict, "paddle_dy")

configs = paddle.SaveLoadConfig()
configs.keep_name_table = True
para_state_dict, _ = paddle.load("paddle_dy", configs)

print(para_state_dict)
# the name_table is 'StructuredToParameterName@@'
# {'bias': array([0.], dtype=float32), 
#  'StructuredToParameterName@@': 
#     {'bias': u'linear_0.b_0', 'weight': u'linear_0.w_0'}, 
#  'weight': array([[ 0.04230034],
#     [-0.1222527 ],
#     [ 0.7392676 ],
#     [-0.8136974 ],
#     [ 0.01211023]], dtype=float32)}
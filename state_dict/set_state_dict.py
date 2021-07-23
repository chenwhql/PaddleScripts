import paddle   

paddle.disable_static()

emb = paddle.nn.Embedding([10, 10])

state_dict = emb.state_dict()
paddle.save(state_dict, "paddle_dy")

para_state_dict, _ = paddle.load("paddle_dy")

emb.set_dict(para_state_dict)
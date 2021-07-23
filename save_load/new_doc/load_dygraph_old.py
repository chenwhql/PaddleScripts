import paddle
import paddle.fluid as fluid

paddle.disable_static()

emb = paddle.nn.Embedding(10, 10)

state_dict = emb.state_dict()
fluid.save_dygraph(state_dict, "paddle_dy")

scheduler = paddle.optimizer.lr_scheduler.NoamLR(	
    d_model=0.01, warmup_steps=100, verbose=True)
adam = paddle.optimizer.Adam(
    learning_rate=scheduler,
    parameters=emb.parameters())
state_dict = adam.state_dict()
print(state_dict)
fluid.save_dygraph(state_dict, "paddle_dy")

para_state_dict, opti_state_dict = fluid.load_dygraph("paddle_dy")
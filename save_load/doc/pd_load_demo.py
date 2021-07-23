import paddle
            
paddle.disable_static()

emb = paddle.nn.Embedding([10, 10])

state_dict = emb.state_dict()
paddle.save(state_dict, "paddle_dy")

scheduler = paddle.optimizer.lr_scheduler.NoamLR(
    d_model=0.01, warmup_steps=100, verbose=True)
adam = paddle.optimizer.Adam(
    learning_rate=scheduler,
    parameters=emb.parameters())
state_dict = adam.state_dict()
paddle.save(state_dict, "paddle_dy")

para_state_dict, opti_state_dict = paddle.load("paddle_dy")
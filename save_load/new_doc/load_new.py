import paddle
            
paddle.disable_static()
emb = paddle.nn.Embedding(10, 10)
layer_state_dict = emb.state_dict()
paddle.save(layer_state_dict, "emb.pdparams")
scheduler = paddle.optimizer.lr_scheduler.NoamLR(	
    d_model=0.01, warmup_steps=100, verbose=True)
adam = paddle.optimizer.Adam(
    learning_rate=scheduler,
    parameters=emb.parameters())
opt_state_dict = adam.state_dict()
paddle.save(opt_state_dict, "adam.pdopt")
load_layer_state_dict = paddle.load("emb.pdparams")
load_opt_state_dict = paddle.load("adam.pdopt")
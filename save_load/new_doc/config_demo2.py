import paddle

# enable dygraph mode
paddle.disable_static() 

# use SaveLoadconfig when loading model
model_path = "simplenet.example.model"
config = paddle.SaveLoadConfig()
config.model_filename = "__simplenet__"
infer_net = paddle.jit.load(model_path, config=config)
# inference
x = paddle.randn([4, 8], 'float32')
pred = infer_net(x)
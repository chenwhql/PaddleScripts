import paddle
import paddle.fluid as fluid
import numpy as np

class Tanspose(paddle.nn.Layer):
    def __init__(self):
        super(Tanspose, self).__init__()

    def forward(self, x0):
        x188 = paddle.fft.rfftn(x=x0, axes=[-2, -1], norm='ortho')
        x189 = paddle.real(x=x188)
        print(x189)
        x190 = paddle.imag(x=x188)
        x191 = [x189, x190]
        x192 = paddle.stack(x=x191, axis=-1)
        x194 = paddle.transpose(x=x192, perm=[0, 1, 4, 2, 3])
        return x194

def main(x0):
    # There are 1 inputs.
    # data: shape-[-1, 3, 300, 300], type-float32.
    paddle.disable_static()
    # params = paddle.load(r'/ssd2/wangjunjie06/pr_for_x2paddle/1222/X2Paddle/CAFFE_MODEL/SSD/pd_model_dygraph/model.pdparams')
    model = Tanspose()
    # model.set_dict(params, use_structured_name=True)
    model.eval()
    # convert to jit
    sepc_list = list()
    sepc_list.append(
            paddle.static.InputSpec(
                shape=[1, 192, 128, 28], name="x0", dtype="float32"))
    static_model = paddle.jit.to_static(model, input_spec=sepc_list)
    paddle.jit.save(static_model, "inference_model/model")
    out = model(x0)
    return out

x0 = np.load("transpose_data.npy")

x0 = paddle.to_tensor(x0)

result = main(x0)
print(result.shape)
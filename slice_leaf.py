import paddle

img = paddle.randn(shape=[100])
img.stop_gradient = False

img[10] = img[10] + 10.0

out = img * 2

rlt = paddle.mean(out)

rlt.backward()


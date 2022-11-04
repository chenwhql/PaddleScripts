import paddle

x = paddle.to_tensor([[[1., 2., 3., 4.],
                        [5., 6., 7., 8.],
                        [9., 10., 11., 12.]],
                      [[13., 14., 15., 16.],
                        [17., 18., 19., 20.],
                        [21., 22., 23., 24.]]])
out1 = paddle.mean(x)
# [12.5]
out2 = paddle.mean(x, axis=-1)
# [[ 2.5  6.5 10.5]
#  [14.5 18.5 22.5]]
out3 = paddle.mean(x, axis=-1, keepdim=True)
# [[[ 2.5]
#   [ 6.5]
#   [10.5]]
#  [[14.5]
#   [18.5]
#   [22.5]]]
out4 = paddle.mean(x, axis=[0, 2])
# [ 8.5 12.5 16.5]
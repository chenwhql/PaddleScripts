import paddle

# paddle.set_device('cpu')

t = paddle.to_tensor([[0.6964692+0.4809319j, 0.28613934+0.39211753j, 0.22685145+0.343178j,
  0.5513148 +0.7290497j],
 [0.71946895+0.43857226j, 0.42310646+0.0596779j, 0.9807642 +0.39804426j,
  0.6848297 +0.7379954j]])
t.stop_gradient = False

out = paddle.abs(t)
print("out:", out)

out.backward()

print("grad: ", t.grad)
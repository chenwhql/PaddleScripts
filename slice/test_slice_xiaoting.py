import paddle
import numpy as np
 
x_pd = paddle.to_tensor(np.array([11, 78, 29, 99, 52, 97, 61, 39, 1, 56]))
x_pd_2d = paddle.randint(high=100, shape=[3, 4])
x_pd_nd = paddle.randint(high=100, shape=[3, 4, 5])
 
# meshgrid
a = paddle.ones([10, 10])
ii = paddle.arange(0, 10, 2)
jj = paddle.arange(0, 10, 2)
xx, yy = paddle.meshgrid(ii, jj)
print('xx shape: ', xx.shape)
b = a[xx, yy]
print('b: ', b)
# 1 int index
# read
print("read int index: ", x_pd[2], x_pd[5], x_pd[8])
# write
x_pd[3] = 100
print("write int index: ", x_pd)
# change value
x_pd[3] -= 5
print("change int index: ", x_pd)

# 2 list index
ind = [3, 4, 6]
print("list index: ", x_pd[ind])

# 3 array index
ind = np.array([3, 4])
# # read
print("read array index: ", x_pd[ind])
# # write
x_pd[ind] = 100
print("write array index: ", x_pd)

# 4 N-D array index
ind = np.array([[3, 7], [4, 5]])
print("N-D array index: ", x_pd[ind])

# 5 row col index
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
print("array row col index: ", x_pd_2d[row, col])
print("int row col index: ", x_pd_2d[2, 1])
# 6 np.newaxis index
print("newaxis index only: ", x_pd_2d[:, np.newaxis])
print("newaxis index: ", x_pd_2d[row[:, np.newaxis], col])

# 7 slice
print("slice: ", x_pd_2d[:, 1:3])

# 8 Ellipsis
print("Ellipsis 1 : ", x_pd_nd[..., 1])
print("Ellipsis 2 : ", x_pd_nd[:, :, 1])

# 9 mask index
mask = np.array([1, 0, 1, 0], dtype=bool)
# print("mask index 1 : ", x_pd_2d[x_pd_2d[1, :], mask])
mask = paddle.to_tensor(mask)
mask_pd = paddle.masked_select(x_pd_2d[1, :], mask)
print("mask index 2 (mask_select): ", mask_pd)

pd_mask = paddle.randn([3, 3]) > 0
xpd = paddle.randn([3, 3])
ypd = paddle.randn([3, 3])
xpd[pd_mask] = ypd[pd_mask]
print("mask index 3 (set value): ", xpd)

# 10 combined index
# print("combined index: ", x_pd_2d[2, [2, 0, 1]])

# 11 n-d tensor index
pred_pd = paddle.arange(240).reshape([10, 6, 4])
index_pd = paddle.arange(5)
pd_result = pred_pd[index_pd, index_pd]
print("n-d tensor index: ", pd_result)
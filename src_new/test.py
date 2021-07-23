
import numpy as np
import paddle.fluid as fluid
import os
import sys
import cv2
import time
import shapely
from shapely.geometry import Polygon


import paddle
from paddle.utils.cpp_extension import load

custom_ops = load(
    name="custom_jit_ops",
    sources=["rbox_iou_op.cc", "rbox_iou_op.cu"])

paddle.set_device('gpu:0')
#paddle.set_device('gpu')
#paddle.set_device('cpu')
paddle.disable_static()

rbox1 = [[772.0, 575.5, 90.75791931152344, 26.0, -0.3468348975810076]]
rbox1 = np.array(rbox1)

rbox2 = [[772.0, 575.5, 90.75791931152344, 26.0, 0.0]]
rbox2 = np.array(rbox2)


use_rand_data = True
# use_rand_data = False

if use_rand_data:
    rbox1 = np.random.rand(13000, 5)
    rbox2 = np.random.rand(7, 5)

    # x1 y1 w h [0, 0.5]
    rbox1[:, 0:4] = rbox1[:, 0:4] * 0.45 + 0.001
    rbox2[:, 0:4] = rbox2[:, 0:4] * 0.45 + 0.001

    rbox1[:, 4] = rbox1[:, 4] - 0.5
    rbox2[:, 4] = rbox2[:, 4] - 0.5


print('rbox1', rbox1.shape, 'rbox2', rbox2.shape)

print('rbox1', rbox1)
print('rbox2', rbox2)


pd_rbox1 = paddle.to_tensor(rbox1, dtype="float32")
pd_rbox2 = paddle.to_tensor(rbox2, dtype="float32")

res = custom_ops.rbox_iou(pd_rbox1, pd_rbox2)
start_time = time.time()
print(pd_rbox1)
print(pd_rbox2)
print('paddle cpu cuda\n\n')
res = custom_ops.rbox_iou(pd_rbox1, pd_rbox2)
print('paddle time:', time.time() - start_time)
print(res.cpu().shape)
print(res.cpu())


def rbox2poly_single(rrect, get_best_begin_point=False):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x_ctr, y_ctr, width, height, angle = rrect[:5]
    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    # rect 2x4
    rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    # poly
    poly = R.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    return poly


def intersection(g, p):
    """
    Intersection.
    """

    g = g[:8].reshape((4, 2))
    p = p[:8].reshape((4, 2))

    a = g
    b = p

    use_filter = True
    if use_filter:
        # step1:
        inter_x1 = np.maximum(np.min(a[:, 0]), np.min(b[:, 0]))
        inter_x2 = np.minimum(np.max(a[:, 0]), np.max(b[:, 0]))
        inter_y1 = np.maximum(np.min(a[:, 1]), np.min(b[:, 1]))
        inter_y2 = np.minimum(np.max(a[:, 1]), np.max(b[:, 1]))
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.
        x1 = np.minimum(np.min(a[:, 0]), np.min(b[:, 0]))
        x2 = np.maximum(np.max(a[:, 0]), np.max(b[:, 0]))
        y1 = np.minimum(np.min(a[:, 1]), np.min(b[:, 1]))
        y2 = np.maximum(np.max(a[:, 1]), np.max(b[:, 1]))
        if x1 >= x2 or y1 >= y2 or (x2 - x1) < 2 or (y2 - y1) < 2:
            return 0.

    g = Polygon(g)
    p = Polygon(p)
    #g = g.buffer(0)
    #p = p.buffer(0)
    if not g.is_valid or not p.is_valid:
        return 0

    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def calc_IoU(a, b):
    """
    Args:
        a:
        b:

    Returns:

    """
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)

    a[:, 2] = a[:, 0] + a[:, 2]
    a[:, 3] = a[:, 1] + a[:, 3]
    b[:, 2] = b[:, 0] + b[:, 2]
    b[:, 3] = b[:, 1] + b[:, 3]
    if np.sum(np.min(a) < 0) > 0 or np.sum(np.min(b) < 0) > 0:
        return 0

    # step1:
    inter_x1 = np.maximum(np.min(a[:, 0]), np.min(b[:, 0]))
    inter_x2 = np.minimum(np.max(a[:, 0]), np.max(b[:, 0]))
    inter_y1 = np.maximum(np.min(a[:, 1]), np.min(b[:, 1]))
    inter_y2 = np.minimum(np.max(a[:, 1]), np.max(b[:, 1]))
    if inter_x1>=inter_x2 or inter_y1>=inter_y2:
        return 0.
    x1 = np.minimum(np.min(a[:, 0]), np.min(b[:, 0]))
    x2 = np.maximum(np.max(a[:, 0]), np.max(b[:, 0]))
    y1 = np.minimum(np.min(a[:, 1]), np.min(b[:, 1]))
    y2 = np.maximum(np.max(a[:, 1]), np.max(b[:, 1]))
    if x1>=x2 or y1>=y2 or (x2-x1)<2 or (y2-y1)<2:
        return 0.
    else:
        mask_w = np.int(np.ceil(x2-x1))
        mask_h = np.int(np.ceil(y2-y1))
        mask_a = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
        mask_b = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
        a[:, 0] -= x1
        a[:, 1] -= y1
        b[:, 0] -= x1
        b[:, 1] -= y1
        print('a', a)
        mask_a = cv2.fillPoly(mask_a, pts=np.asarray(a, 'int32'), color=1)
        mask_b = cv2.fillPoly(mask_b, pts=np.asarray(b, 'int32'), color=1)
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        iou = float(inter)/(float(union)+1e-12)
        # print(iou)
        # cv2.imshow('img1', np.uint8(mask_a*255))
        # cv2.imshow('img2', np.uint8(mask_b*255))
        # k = cv2.waitKey(0)
        # if k==ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()
        return iou


def rbox_overlaps(anchors, gt_bboxes, use_cv2=False):
    """

    Args:
        anchors: [NA, 5]  x1,y1,x2,y2,angle
        gt_bboxes: [M, 5]  x1,y1,x2,y2,angle

    Returns:

    """
    assert anchors.shape[1] == 5
    assert gt_bboxes.shape[1] == 5

    gt_bboxes_ploy = [rbox2poly_single(e) for e in gt_bboxes]
    anchors_ploy = [rbox2poly_single(e) for e in anchors]

    num_gt, num_anchors = len(gt_bboxes_ploy), len(anchors_ploy)
    iou = np.zeros((num_gt, num_anchors), dtype=np.float32)

    if use_cv2:
        for i in range(num_gt):
            for j in range(num_anchors):
                try:
                    iou[i, j] = calc_IoU(gt_bboxes_ploy[i].astype(np.int32), anchors_ploy[j].astype(np.int32))
                except Exception as e:
                    print('cur calc_IoU', e)
        iou = iou.T
        return iou

    start_time = time.time()
    for i in range(num_gt):
        for j in range(num_anchors):
            try:

                iou[i, j] = intersection(gt_bboxes_ploy[i], anchors_ploy[j])

            except Exception as e:
                print('cur gt_bboxes_ploy[i]', gt_bboxes_ploy[i], 'anchors_ploy[j]', anchors_ploy[j], e)
    iou = iou.T
    print('intersection  all sp_time', time.time() - start_time)
    return iou


ploy_rbox1 = rbox1
ploy_rbox1[:, 0:4] = ploy_rbox1[:, 0:4] * 1024
ploy_rbox2 = rbox2
ploy_rbox2[:, 0:4] = ploy_rbox2[:, 0:4] * 1024

start_time = time.time()
iou1 = rbox_overlaps(ploy_rbox1, ploy_rbox2, use_cv2=False)
print('rbox time', time.time() - start_time)
print(iou1.shape)
print(iou1)

res1 = res.cpu().numpy()
print('iou1', iou1.shape, 'res1', res1.shape)
print('diff', np.sum(np.abs(iou1 - res1)))
diff = (iou1 - res1) / (res1 + 1e-8)
print('diff1 shape', diff.shape, 'abs sum', np.abs(diff).sum(), 'abs mean', np.abs(diff).mean(), 'abs max', np.abs(diff).max())
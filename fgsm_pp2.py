#encoding:utf-8
import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import Compose, Normalize

import numpy as np
import cv2

#对比展现原始图片和对抗样本图片
def show_images_diff(original_img,original_label,adversarial_img,adversarial_label):
    import matplotlib.pyplot as plt
    plt.figure()

    #归一化
    if original_img.any() > 1.0:
        original_img=original_img/255.0
    if adversarial_img.any() > 1.0:
        adversarial_img=adversarial_img/255.0

    plt.subplot(131)
    plt.title('Original')
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adversarial') 
    plt.imshow(adversarial_img)
    plt.axis('off')

    plt.subplot(133)
    plt.title('Adversarial-Original')
    difference = adversarial_img - original_img
    print ("diff shape: ", difference.shape)
    #(-1,1)  -> (0,1)
    difference=difference / abs(difference).max()/2.0+0.5
    plt.imshow(difference,cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


#获取计算设备 默认是CPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#图像加载以及预处理
image_path="./picture/cropped_panda.jpg"
orig = cv2.imread(image_path)[..., ::-1]
orig = cv2.resize(orig, (224, 224))
img = orig.copy().astype(np.float32)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img /= 255.0
img = (img - mean) / std
img = img.transpose(2, 0, 1)

img = np.expand_dims(img, axis=0)
img = paddle.to_tensor(img, dtype='float32', place=paddle.CUDAPlace(0), stop_gradient=False)
print (img.stop_gradient)
print(img.shape)

#使用预测模式 主要影响droupout和BN层的行为
#model = paddle.vision.models.resnet50(pretrained=True) #效果不对，483，388是才是pandda 
model = paddle.vision.models.vgg16(pretrained=True)
predict = model(img)[0]
print (predict.shape)
label = np.argmax(predict)
print("label={}".format(label)) #388

#设置为不保存梯度值 自然也无法修改
for param in model.parameters():
    param.stop_gradient = True
    
#optimizer = torch.optim.Adam([img])
# 设置优化器
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=img)
#loss_func = paddle.nn.CrossEntropyLoss()


epochs=100
target=288
target = paddle.to_tensor(target, dtype='int64',place=paddle.CUDAPlace(0))

for epoch in range(epochs):
    # forward + backward
    output = model(img)
    #loss = loss_func(output, target)

    loss = F.cross_entropy(output, target)
    label = np.argmax(output[0])
    #print("label={}".format(label)) 
    print("epoch={} loss={} label={}".format(epoch,loss,label))
    
    #如果定向攻击成功
    if label == target:
        break  
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()


#todo
adv = img.data.cpu().numpy()[0]
print(adv.shape)
print(adv)
print(type(adv))

adv = adv.transpose(1, 2, 0)
adv = (adv * std) + mean
adv = adv * 255.0
#adv = adv[..., ::-1]  # RGB to BGR
adv = np.clip(adv, 0, 255).astype(np.uint8)

show_images_diff(orig,388,adv,target.data.cpu().numpy()[0])

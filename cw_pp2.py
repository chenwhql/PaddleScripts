#encoding:utf-8

#CW
import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import Compose, Normalize

import numpy as np
import cv2

def show_images_diff(original_img,original_label,adversarial_img,adversarial_label):
    import matplotlib.pyplot as plt
    plt.figure()
        
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
    cv2.imwrite("output/cw_diff.png", difference)

    l0=np.where(difference!=0)[0].shape[0]
    l2=np.linalg.norm(difference)
    #print(difference)
    print("l0={} l2={}".format(l0, l2))
    
    #(-1,1)  -> (0,1)
    difference=difference / abs(difference).max()/2.0+0.5
    
    plt.imshow(difference,cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    plt.show()



#获取计算设备 默认是CPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paddle.set_device('cpu')

#图像加载以及预处理
image_path="./picture/cow.jpeg"
# image_path="./picture/cropped_panda.jpg"
orig = cv2.imread(image_path)[..., ::-1]
print ("orig shape -1: ", orig.shape)#HWC
orig = cv2.resize(orig, (224, 224))
img = orig.copy().astype(np.float32)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img /= 255.0
img = (img - mean) / std
img = img.transpose(2, 0, 1)

img = np.expand_dims(img, axis=0)
print(img.shape)

#使用预测模式 主要影响droupout和BN层的行为
model = paddle.vision.models.vgg16(pretrained=True)
model_s = paddle.Model(model)
print(model_s.summary((-1, 3, 224, 224)))
tmp = model.features(paddle.to_tensor(img, dtype="float32"))
#tmp = model.features[0](paddle.to_tensor(img, dtype="float32"))
#print(tmp.shape)
#tmp = model.features[4](paddle.to_tensor(img, dtype="float32"))
#print(tmp.shape)
##todo avg_pool
tmp = paddle.flatten(tmp, 1)
fc3 = model.classifier(tmp)
#print("Linear-3: ", fc3)
print("Linear-3: ", fc3.shape, type(fc3)) #[1, 1000]

#adam的最大迭代次数 论文中建议10000次 测试阶段1000也可以 1000次可以完成95%的优化工作
#max_iterations = 1000
max_iterations = 100 #mod by zh
#adam学习速率
learning_rate = 0.01
#二分查找最大次数
binary_search_steps = 10
#c的初始值
initial_const = 1e2
confidence = initial_const

#k值
k = 40

#像素值区间
boxmin = -3.0
boxmax = 3.0

#类别数 pytorch的实现里面是1000
num_labels = 1000

#攻击目标标签 必须使用one hot编码
#target_label = 288
target_label = 344
tlab = paddle.eye(num_labels)[target_label]
print("type of tlab: ", type(tlab))


print()

shape = (1,3,224,224)


#c的初始化边界
lower_bound = 0
c = initial_const
upper_bound = 1e10

# the best l2, score, and image attack
o_bestl2 = 1e10
o_bestscore = -1
o_bestattack = [np.zeros(shape)]

# the resulting image, tanh'd to keep bounded from boxmin to boxmax
boxmul = (boxmax - boxmin) / 2.
boxplus = (boxmin + boxmax) / 2.

# output = model(paddle.to_tensor(img, dtype="float32", place=paddle.CUDAPlace(0)))
output = model(paddle.to_tensor(img, dtype="float32"))
orig_label = np.argmax(output)
print("orig_label={}".format(orig_label), type(orig_label))  #345

succ_flag = False
for outer_step in range(binary_search_steps):
    print("o_bestl2={} confidence={}".format(o_bestl2, confidence)  )
    
    #把原始图像转换成图像数据和扰动的形态
    timg = paddle.to_tensor(np.arctanh((img - boxplus) / boxmul * 0.999999), dtype='float32')
    modifier = paddle.zeros_like(timg, dtype='float32')
    #print (type(modifier))
    # modifier = paddle.to_tensor(modifier, dtype='float32', place=paddle.CUDAPlace(0))
    #print (type(modifier))
    #图像数据的扰动量梯度可以获取
    import pdb
    pdb.set_trace()
    modifier.stop_gradient = False

    #设置为不保存梯度值 自然也无法修改
    for param in model.parameters():
        param.stop_gradient = True
        
    #定义优化器 仅优化modifier
    #optimizer = torch.optim.Adam([modifier], lr=learning_rate)
    optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=[modifier])
    
    for iteration in range(1, max_iterations + 1):
        optimizer.clear_grad()
        
        #定义新输入 
        newimg = paddle.tanh(modifier + timg) * boxmul + boxplus
        #print ("newimg shape: ", newimg.shape)       #[1, 3, 224, 224]
        #print ("new img type: ", type(newimg))      
        #newimg = paddle.to_tensor(newimg)
        #print ("new img type: ", type(newimg))      
        output = model(newimg)
        pred_label = np.argmax(output)
        print("in iter pred_label={}".format(pred_label))  #345
        #定义cw中的损失函数
        
        #l2范数
        # print(newimg)
        # print(paddle.tanh(timg) * boxmul + boxplus)
        # loss2 = paddle.dist(newimg, (paddle.tanh(timg) * boxmul + boxplus), p=2)
        loss2 = paddle.sum((newimg - (paddle.tanh(timg) * boxmul + boxplus)) ** 2)
        
        """
        # compute the probability of the label class versus the maximum other
            real = tf.reduce_sum((tlab)*output,1)
            # 论文中的开源实现 
            #other = tf.reduce_max((1-tlab)*output - (tlab*10000),1)
            other = tf.reduce_max((1-tlab)*output)
            loss1 = tf.maximum(0.0, other-real+k)
            loss1 = tf.reduce_sum(const*loss1)
        """
        #paddle.max((1 - tlab) * output) -  paddle.max(output * tlab) +k
        real = paddle.max(output * tlab)
        other = paddle.max((1 - tlab) * output)  
        loss1 = other - real + k   
        loss1 = paddle.clip(loss1, min=0)
             
        loss1 = confidence * loss1
           
        loss = loss1 + loss2
            
        loss.backward(retain_graph=True)
        pdb.set_trace()
        print(modifier.grad)
        optimizer.step()

        print(modifier)
              
        l2 = loss2
        
        sc = output
        print("in loss pred_label: ", np.argmax(sc.numpy()))
        
        # print out the losses every 10%
        if iteration%(max_iterations//10) == 0:
            print("iteration={} loss={} loss1={} loss2={}".format(iteration,loss,loss1,loss2))
              
        #if (l2 < o_bestl2) and (np.argmax(sc) == target_label):
        if (l2 < o_bestl2) and (np.argmax(sc.numpy()) != orig_label):
            print("attack success l2={} target_label={} pred_label={}".format(l2, target_label, np.argmax(sc.numpy())))
            o_bestl2 = l2
            o_bestscore = np.argmax(sc.numpy())
            #o_bestattack = newimg.data.cpu().numpy()
            o_bestattack = newimg
            #<class 'paddle.VarBase'> [1, 3, 224, 224]
            print("o_bestattack type: ", type(o_bestattack), o_bestattack.shape) 
            
            ###
            #adv = o_bestattack[0]
            adv = o_bestattack.numpy()
            print("o_bestattack type: ", type(o_bestattack)) #<class 'paddle.VarBase'>
            print("====adv shape: ", adv.shape, type(adv)) # [1, 3, 224, 224]
            adv = np.squeeze(adv)
            print("====after squeeze adv shape: ", adv.shape)#[3, 224, 224]
            adv = adv.transpose(1, 2, 0)
            adv = (adv * std) + mean
            adv = adv * 255.0
            adv = np.clip(adv, 0, 255).astype(np.uint8)
            cv2.imwrite("output/cw_adv_{}_{}.png".format(outer_step, iteration), adv )

            show_images_diff(orig, 0, adv, 0)
            ###
            #succ_flag = True
            #break # mod by zh

    #mod by zh 
    #if succ_flag:
    #    break
    confidence_old = -1       
    if (o_bestscore == target_label) and o_bestscore != -1:
        #攻击成功 减小c
        upper_bound = min(upper_bound, confidence)
        if upper_bound < 1e9:
                print()
                confidence_old = confidence
                confidence = (lower_bound + upper_bound)/2
    else:
        lower_bound = max(lower_bound, confidence)
        confidence_old = confidence
        if upper_bound < 1e9:
                confidence = (lower_bound + upper_bound) / 2
        else:
                confidence *= 10
                
    print("outer_step={} confidence {}->{}".format(outer_step, confidence_old, confidence))



print(o_bestattack.shape)
print(img.shape)


adv = o_bestattack.numpy()
#adv=o_bestattack[0]
print(adv.shape)
adv = np.squeeze(adv)
adv = adv.transpose(1, 2, 0)
adv = (adv * std) + mean
adv = adv * 255.0
adv = np.clip(adv, 0, 255).astype(np.uint8)
cv2.imwrite("output/cw_adv.png", adv )

show_images_diff(orig, 0, adv, 0)

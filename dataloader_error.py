import paddle as pdl
from paddle.vision.models import resnet50
from paddle.io import Dataset, IterableDataset
import numpy as np

pdl.seed(32)

class Reader(IterableDataset):
    def __init__(self, dataset):
        super(Reader, self).__init__()
        self.dataset = dataset

    def __iter__(self):
        for sample in self.dataset:
            im, lab = sample
            im = im.resize((256, 256))
            im = np.array(im).astype("float32").transpose((2, 1, 0)) / 255
            yield im, lab
# class Reader(Dataset):
#     def __init__(self, dataset):
#         super(Reader, self).__init__()
#         self.dataset = dataset

#     def __getitem__(self, item):
#         im, lab = self.dataset[item]
#         im = im.resize((256, 256))
#         im = np.array(im).astype("float32").transpose((2, 1, 0)) / 255
#         return im, lab

#     def __len__(self):
#         return len(self.dataset)


train_datasets = pdl.vision.datasets.Flowers(mode='train')
val_datasets = pdl.vision.datasets.Flowers(mode='test')
train_reader = Reader(train_datasets)
val_reader = Reader(val_datasets)
train_reader = pdl.io.DataLoader(train_reader, batch_size=1, num_workers=0, use_shared_memory=True)
val_reader = pdl.io.DataLoader(val_reader, batch_size=1, num_workers=4, use_shared_memory=False)

net = pdl.nn.Sequential(resnet50(pretrained=True, num_classes=1000, with_pool=True),
                        pdl.nn.Linear(1000, 102))

# 定于输入层
input_define = pdl.static.InputSpec(shape=[-1, 3, 256, 256],
                                    dtype="float32",
                                    name="img")
# 定义Label输入层，用于计算损失和准确率
label_define = pdl.static.InputSpec(shape=[-1, 1],
                                    dtype="int64",
                                    name="label")
# 实例化网络对象并定义优化器等训练逻辑
model = pdl.Model(net, inputs=input_define, labels=label_define)
# 此处使用SGD优化器，可尝试使用Adam，收敛效果更好更快速
optimizer = pdl.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
# 损失函数使用交叉熵，评价指标使用准确率
# 其中Top-k表示推理得到的概率分布中，概率最高的前k个推理结果中是否包含正确标签，如果包含则视为正确，这里的1，2，3分别计算k为1~3的情况
model.prepare(optimizer=optimizer,
              loss=pdl.nn.CrossEntropyLoss(),
              metrics=pdl.metric.Accuracy(topk=(1, 5)))

vdl = pdl.callbacks.VisualDL("./log_BenchMark_4_102")

model.fit(train_data=train_reader,
          eval_data=val_reader,
          batch_size=128,
          epochs=100,
          save_dir="output/",
          save_freq=10,
          log_freq=1000,
          callbacks=vdl,
          num_workers=0)
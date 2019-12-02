#!/usr/bin/env python
# coding=utf-8
# encoding=utf8

import paddle.fluid as fluid
from paddle.dataset import imdb
from paddle.fluid import layers
from paddle.fluid import nets

import numpy as np

def textcnn(token_ids, vocab_size, num_classes, emb_dim, num_filters, mlp_hid_dim):
    """TextCNN模型前向过程实现
    Args:
        token_ids: 包含不同长度样本的lod tensor，形状为[-1,1]
        vocab_size: 词典大小
        num_classes: 类别数量
        emb_dim: 词向量维度
        num_filters: 每种尺寸的卷积核数量
        mlm_hid_dim: MLP的隐层维度
    Returns:
        prediction: 预测结果，各个类别的概率分布
    """
    emb = layers.embedding(               # 得到输入样本的词向量表示
        input=token_ids, size=[vocab_size, emb_dim])

    res_size3 = nets.sequence_conv_pool(   # 尺寸为3的卷积层&池化操作
        input=emb,
        num_filters=num_filters,
        filter_size=3,
        act="tanh",
        pool_type="max")
    res_size4 = nets.sequence_conv_pool(      # 尺寸为4的卷积层&池化操作
        input=emb,
        num_filters=num_filters,
        filter_size=4,
        act="tanh",
        pool_type="max")
    res_size5 = nets.sequence_conv_pool(   # 尺寸为5的卷积层&池化操作
        input=emb,
        num_filters=num_filters,
        filter_size=5,
        act="tanh",
        pool_type="max")
    hidden = layers.fc(                   # 特征向量到MLP隐层的映射
        input=[res_size3, res_size4, res_size4], size=mlp_hid_dim)
    prediction = fluid.layers.fc(                 # MLP隐层到类别
        input=hidden, size=num_classes, act="softmax")
    return prediction


def build_data_layer(inputs_generator_fn):
    """ 异步数据读取"""

    x = layers.data('token_ids', shape=[-1, 1], dtype='int64')
    y = layers.data('labels', shape=[-1, 1], dtype='int64')

    reader = fluid.io.PyReader([x, y], capacity=1, iterable=False)
    reader.decorate_batch_generator(inputs_generator_fn)
    reader.start()

    return x, y


def build_train_program(conf, data_gen_fn):
    """创建模型的训练program
       Args:
           - conf: 配置字典，包含模型的超参数
           - data_gen_fn: 样本生成器函数，用于读取训练样本
       return:
           - init_prog: 初始化程序，用于模型初始化等
           - train_prog: 训练程序，用于模型训练
           - fetch_list: 需要在训练过程中取出的变量，用于可视化、后处理等。注意paddlepaddle中运行的基本单位是program，不会根据fetch_list进行计算图剪枝"""

    train_prog = fluid.Program()
    init_prog = fluid.Program()
    with fluid.program_guard(train_prog, init_prog):
        x, y = build_data_layer(data_gen_fn)
        prediction = textcnn(x, conf['vocab_size'], conf['num_classes'], conf['emb_dim'], \
                             conf['num_filters'], conf['mlp_hid_dim'])
        loss = fluid.layers.cross_entropy(input=prediction, label=y)
        loss = fluid.layers.mean(loss)
        accuracy = fluid.layers.accuracy(input=prediction, label=y)
        adam_optimizer = fluid.optimizer.Adam(learning_rate=0.001)
        adam_optimizer.minimize(loss)

    fetch_list = [loss, accuracy]
    return init_prog, train_prog, fetch_list



def array_normalize(x, dtype=None, expand_dims=True, return_lod_tensor=False):
    """输入归一化函数，将输入转为numpy或lod tensor并扩展一个维度
        Args:
            - x: 需要被归一化的输入，list或numpy类型
            - dtype: 需要被归一化到的数据类型，整型数据一般使用'int64'，浮点型数据一般使用'float32'，此外还支持'int8', 'int16', int32', 'float16'和'bool'类型
            - expand_dims: 扩展一个维度，默认开启。如shape为[batch_size]的输入会转为[batch_size, 1]的shape
            - return_lod_tensor: 将输入转为lod tensor的类型，默认关闭。否则转为numpy类型。
        Returns:
            - x: 对输入x归一化后的结果，numpy或lod tensor类型"""
    if return_lod_tensor:
        lens = [len(i) for i in x]
        x = [i for j in x for i in j]
        x = np.array(x)
        if dtype is not None:
            x = x.astype(dtype)
        if expand_dims:
            x = np.expand_dims(x, -1)
        x = fluid.create_lod_tensor(x, recursive_seq_lens=[lens], place=fluid.CPUPlace())
    else:
        assert np.ndim(x) < 2 or len(set([len(i) for i in x])) == 1, "lengths of elements on dim1 are inconsistent, please pad them to the same length or set return_lod_tensor=True."
        x = np.array(x)
        if dtype is not None:
            x = x.astype(dtype)
        if expand_dims:
            x = np.expand_dims(x, -1)
    return x


def build_executor(use_gpu=False):
    if use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    return fluid.Executor(place)


def create_data_generator(word_dict, batch_size, num_epochs=1, is_train=False):
    """样本生成器创建函数，用于创建样本生成器。
        Args: 
            - batch_size: 训练和推理时传入网络的batch的大小
            - num_epochs: 对数据集的遍历次数
            - is_train: 训练/推理标志位
        Return:
            - data_generator_fn: 样本生成器函数"""

    if is_train:
        examples = [i for i in imdb.train(word_dict)()]
        np.random.shuffle(examples)
    else:
        examples = [i for i in imdb.test(word_dict)()]

    def data_generator_fn():
        batch_x = []
        batch_y = []
        for i in range(num_epochs):
            print('Training epoch {}:'.format(i))
            for _x, _y in examples:
                # 为了避免遭遇过长样本导致显存溢出，我们将句子长度截断到800
                batch_x.append(_x[:800])
                batch_y.append(_y)
                if len(batch_x) == batch_size:
                    batch_x = array_normalize(batch_x, return_lod_tensor=False)
                    batch_y = array_normalize(batch_y)
                    yield [batch_x, batch_y]
                    batch_x = []
                    batch_y = []

    return data_generator_fn


if __name__ == '__main__':

    conf = {
        'num_epochs': 2,
        'batch_size': 64,
        'num_classes': 2,
        'emb_dim': 128,
        'num_filters': 128,
        'mlp_hid_dim': 128
    }

    word_dict = imdb.word_dict()
    conf['vocab_size'] = len(word_dict)

    # 建立训练数据集的样本生成器
    train_generator_fn = create_data_generator(word_dict, conf['batch_size'], conf['num_epochs'], is_train=True)

    # 建立训练program
    init_prog, train_prog, fetch_list = build_train_program(conf, train_generator_fn)

    # 建立执行器
    exe = build_executor(use_gpu=True)

    # 模型初始化
    exe.run(init_prog)

    # 模型训练
    steps = 0
    try: 
        while True:
            steps += 1
            loss, acc = exe.run(train_prog, fetch_list=fetch_list)
            if steps % 20 == 0:
                print("step {}, loss {}, acc {}.".format(steps, loss, acc))
        print(loss)
    except fluid.core.EOFException as e:
        fluid.io.save_params(exe, './output/params', train_prog)
        print('model saved at ./output/params.')



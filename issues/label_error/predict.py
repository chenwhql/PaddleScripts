# -*- coding: utf-8 -*-
import numpy as np
import paddle
import paddle.fluid as fluid

paddle.enable_static()

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

save_path = './infer_model/'

# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)


# 获取数据
def get_data(sentence):
    # 读取数据字典
    with open('./dict_txt.txt', 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])
    dict_txt = dict(dict_txt)
    # 把字符串数据转换成列表数据
    keys = dict_txt.keys()
    data = []
    for s in sentence:
        # 判断是否存在未知字符
        if not s in keys:
            s = '<unk>'
        data.append(int(dict_txt[s]))
    return np.array(data, dtype=np.int64)
    # return data

right = 0
wrong = 0
fw = open('result.tsv', 'w', encoding='utf-8')
with open('question.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        data = [get_data(line.strip('\n').split('\t')[1])]

        # 获取每句话的单词数量
        base_shape = [[len(c) for c in data]]

        # 生成预测数据
        tensor_words = fluid.create_lod_tensor(data, base_shape, place)

        # 执行预测
        result = exe.run(program=infer_program,
                         feed={feeded_var_names[0]: tensor_words},
                         fetch_list=target_var)

        # 获取结果概率最大的label
        for i in range(len(data)):
            lab = np.argsort(result)[0][i][-1]
            fw.write(line.strip('\n') + '\t' + str(lab) + '\n')
            # print('预测结果标签为：%d， 概率为：%f' % (lab, result[0][i][lab]))
            print('预测结果标签为：%d， 实际为：%d' % (lab, int(line.split('\t')[0])))
            if lab == int(line.split('\t')[0]):
                right += 1
            else:
                wrong += 1
fw.close()
print('right:{}'.format(right))
print('wrong:{}'.format(wrong))

print('acc:{}'.format(right / (right + wrong)))

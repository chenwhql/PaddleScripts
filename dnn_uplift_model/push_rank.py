#!/bin/env python
# -*- encoding:utf-8 -*-
"""
push dnn model
Authors: zhangtai jianghanmin
refer to: Deep Title Understanding Network (DTUN)  more info to : http://wiki.baidu.com/pages/viewpage.action?pageId=1136034407
"""
import math
import sys
import os
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.layers as layers
import json
import paddle.fluid.nets as nets

from paddle.fluid.executor import Executor
from paddle.fluid.optimizer import AdamOptimizer
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

USE_GPU = False
BATCH_SIZE = 1024
EMB_LEN = 16
NUM_FILTERS = 32
PASS_NUM = 1
CPU_NUM = 8
cluster_train_dir = "./train_data"
cluster_train_part_dir = "./train_part_data"
cluster_test_dir = "./test_data"
os.environ['CPU_NUM'] = str(CPU_NUM)


def count_discrete(count_num):
    """
    count discrete -> 6 id
    """
    if count_num <= 0:
        discrete_res = 0
    elif count_num == 1:
        discrete_res = 1
    elif count_num >= 2 and count_num <= 3:
        discrete_res = 2
    elif count_num > 3 and count_num <= 5:
        discrete_res = 3
    elif count_num > 5 and count_num <= 10:
        discrete_res = 4
    else:
        discrete_res = 5

    return discrete_res


def score_discrete(score):
    """
    -1~1 score discrete -> 20 id
    """
    score += 1
    if score <= 0:
        discrete_res = 0
    elif score >= 2:
        discrete_res = 20
    else:
        discrete_res = score / 0.1

    return int(discrete_res)


def pctr_discrete(pctr, brand_type):
    """
    0-1 pctr discrete -> 48 id
    """
    if brand_type == "huawei":
        if pctr <= 1.5E-4:
            discrete_res = 0
        elif 1.5E-4 < pctr and pctr <= 2E-4:
            discrete_res = 1
        elif 2E-4 < pctr and pctr <= 2.5E-4:
            discrete_res = 2
        elif 2.5E-4 < pctr and pctr <= 3E-4:
            discrete_res = 3
        elif 3E-4 < pctr and pctr <= 3.5E-4:
            discrete_res = 4
        elif 3.5E-4 < pctr and pctr <= 4E-4:
            discrete_res = 5
        elif 4E-4 < pctr and pctr <= 4.5E-4:
            discrete_res = 6
        elif 4.5E-4 < pctr and pctr <= 5E-4:
            discrete_res = 7
        elif 5E-4 < pctr and pctr <= 5.5E-4:
            discrete_res = 8
        elif 5.5E-4 < pctr and pctr <= 6E-4:
            discrete_res = 9
        elif 6E-4 < pctr and pctr <= 7E-4:
            discrete_res = 10
        elif 7E-4 < pctr and pctr <= 8E-4:
            discrete_res = 11
        elif 8E-4 < pctr and pctr <= 9E-4:
            discrete_res = 12
        elif 9E-4 < pctr and pctr <= 10E-4:
            discrete_res = 13
        elif 10E-4 < pctr and pctr <= 11E-4:
            discrete_res = 14
        elif 11E-4 < pctr and pctr <= 12E-4:
            discrete_res = 15
        elif 12E-4 < pctr and pctr <= 14E-4:
            discrete_res = 16
        elif 14E-4 < pctr and pctr <= 16E-4:
            discrete_res = 17
        elif 16E-4 < pctr and pctr <= 20E-4:
            discrete_res = 18
        elif 20E-4 < pctr and pctr <= 25E-4:
            discrete_res = 19
        elif 25E-4 < pctr and pctr <= 30E-4:
            discrete_res = 20
        elif 30E-4 < pctr and pctr <= 40E-4:
            discrete_res = 21
        else:  # 40E-4 < pctr:
            discrete_res = 22
    elif brand_type == "iphone":
        if pctr <= 4E-4:
            discrete_res = 23
        elif 4E-4 < pctr and pctr <= 5E-4:
            discrete_res = 24
        elif 5E-4 < pctr and pctr <= 6E-4:
            discrete_res = 25
        elif 6E-4 < pctr and pctr <= 7E-4:
            discrete_res = 26
        elif 7E-4 < pctr and pctr <= 8E-4:
            discrete_res = 27
        elif 8E-4 < pctr and pctr <= 9E-4:
            discrete_res = 28
        elif 9E-4 < pctr and pctr <= 10E-4:
            discrete_res = 29
        elif 10E-4 < pctr and pctr <= 11E-4:
            discrete_res = 30
        elif 11E-4 < pctr and pctr <= 12E-4:
            discrete_res = 31
        elif 12E-4 < pctr and pctr <= 14E-4:
            discrete_res = 32
        elif 14E-4 < pctr and pctr <= 16E-4:
            discrete_res = 33
        elif 16E-4 < pctr and pctr <= 20E-4:
            discrete_res = 34
        elif 20E-4 < pctr and pctr <= 25E-4:
            discrete_res = 35
        elif 25E-4 < pctr and pctr <= 30E-4:
            discrete_res = 36
        elif 30E-4 < pctr and pctr <= 40E-4:
            discrete_res = 37
        elif 40E-4 < pctr and pctr <= 50E-4:
            discrete_res = 38
        else:  # 50E-4 < pctr
            discrete_res = 39
    else:  # brand_type == "nohuawei":
        if pctr <= 0.25E-4:
            discrete_res = 40
        elif 0.25E-4 < pctr and pctr <= 0.5E-4:
            discrete_res = 41
        elif 0.5E-4 < pctr and pctr <= 0.75E-4:
            discrete_res = 42
        elif 0.75E-4 < pctr and pctr <= 1E-4:
            discrete_res = 43
        elif 1E-4 < pctr and pctr <= 1.25E-4:
            discrete_res = 44
        elif 1.25E-4 < pctr and pctr <= 1.5E-4:
            discrete_res = 45
        elif 1.5E-4 < pctr and pctr <= 1.75E-4:
            discrete_res = 46
        elif 1.75E-4 < pctr and pctr <= 2E-4:
            discrete_res = 47
        elif 2E-4 < pctr and pctr <= 2.5E-4:
            discrete_res = 48
        elif 2.5E-4 < pctr and pctr <= 3E-4:
            discrete_res = 49
        elif 3E-4 < pctr and pctr <= 4E-4:
            discrete_res = 50
        elif 4E-4 < pctr and pctr <= 6E-4:
            discrete_res = 51
        elif 6E-4 < pctr and pctr <= 10E-4:
            discrete_res = 52
        else:  # 10E-4 < pctr : todo：默认一个大的是不是有风险
            discrete_res = 53

    return discrete_res


def click_title_times_discrete(push_click_times):
    """
        0-n push_click_times discrete -> 5id
    """
    if push_click_times == 0:
        push_click_emb_num = 0
    elif push_click_times == 1:
        push_click_emb_num = 1
    elif 2 <= push_click_times <= 5:
        push_click_emb_num = 2
    elif 5 < push_click_times <= 10:
        push_click_emb_num = 3
    else:  # push_click_times > 10
        push_click_emb_num = 4

    return push_click_emb_num


def set_zero(place, scope, var_name):
    """
    zet zero
    """
    param = scope.var(var_name).get_tensor()
    param_array = np.zeros(param._get_dims()).astype("int64")
    param.set(param_array, place)


def del_sample_same(line):
    """
    deal input line to feature
    """
    arr = line.strip().split('\t')
    user_feature_serialize = [int(x) for x in arr[0].split(';')]
    candidate_feature_serialize = arr[1].split(';')
    context_serialize = [int(x) for x in arr[2].split(';')]
    label = [[int(arr[3])]]
    candidate_title_serialize = arr[8].split(';')[1:]
    click_times = [click_title_times_discrete(int(arr[9]))]
    #  list index out of range 临时处理
    if len(arr) < 11:
        list1 = []
        for i in range(int(arr[9])):
            str1 = "noMatchClick;0;0;0;0"
            list1.append(str1)
        str2 = "#&".join(list1)
        arr.append(str2)
    click_history = [x for x in arr[10].split('#&')]
    if len(click_history) > 29:
        # 最多保留 28条记录
        click_title_serialize = [y.split(';')[1:] for y in click_history[0:28]]
    else:
        click_title_serialize = [y.split(';')[1:] for y in click_history]

    factory_code = user_feature_serialize[0]
    sex = user_feature_serialize[1]
    age = user_feature_serialize[2]
    car_owner = user_feature_serialize[3]
    occupation = user_feature_serialize[4]
    catering_expense_level = user_feature_serialize[5]
    income_level = user_feature_serialize[6]
    up_info = [factory_code, sex, age, car_owner,
               occupation, catering_expense_level, income_level]
    city_id = [user_feature_serialize[7]]
    user_click_id = [user_feature_serialize[8]]
    user_b_click_id = [user_feature_serialize[9]]
    user_c_click_id = [user_feature_serialize[10]]
    user_d_click_id = [user_feature_serialize[11]]

    week = [context_serialize[0]]
    hour = [context_serialize[1]]

    b_c_d = [int(candidate_feature_serialize[0])]
    tags = [[int(x) for x in candidate_feature_serialize[1].split(',')]]
    subtags = [[int(x) for x in candidate_feature_serialize[2].split(',')]]
    tag1_click_id = [int(candidate_feature_serialize[3])]
    subtag1_click_id = [int(candidate_feature_serialize[4])]
    pctr_discrete_id = [int(candidate_feature_serialize[5])]
    # dnn_score_discrete_id = [int(candidate_feature_serialize[6])]
    pctr = [float(candidate_feature_serialize[7])]
    # score = [float(candidate_feature_serialize[8])]
    # content_emb = [[float(x) for x in candidate_feature_serialize[9].split(',')]]
    # user_emb = [[float(x) for x in candidate_feature_serialize[10].split(',')]]
    user_tags_click = [[int(x) for x in candidate_feature_serialize[11].split(',')]]
    user_subtags_click = [[int(x) for x in candidate_feature_serialize[12].split(',')]]

    candidate_title = [[int(x) for x in candidate_title_serialize[0].split(',')]]
    candidate_subtitle = [[int(x) for x in candidate_title_serialize[1].split(',')]]
    candidate_title_len = [int(candidate_title_serialize[2])]
    candidate_subtitle_len = [int(candidate_title_serialize[3])]

    click_title_tmp = [click_title_serialize[i][0] for i in range(0, len(click_title_serialize))]
    click_title = [x.split(',') for x in click_title_tmp]
    click_title_list = []
    for title_id in click_title:
        title_tmp = map(int, title_id)
        click_title_list.append(title_tmp)

    click_subtitle_tmp = [click_title_serialize[i][1] for i in range(0, len(click_title_serialize))]
    click_subtitle = [x.split(',') for x in click_subtitle_tmp]
    click_subtitle_list = []
    for subtitle_id in click_subtitle:
        subtitle_tmp = map(int, subtitle_id)
        click_subtitle_list.append(subtitle_tmp)

    click_title_len_list = [[int(y)] for y in
                            [click_title_serialize[i][2] for i in range(0, len(click_title_serialize))]]
    click_subtitle_len_list = [[int(y)] for y in
                               [click_title_serialize[i][3] for i in range(0, len(click_title_serialize))]]

    # + dnn_score_discrete_id + score + content_emb + user_emb
    return up_info + city_id + \
           user_click_id + user_b_click_id + user_c_click_id + user_d_click_id + \
           week + hour + b_c_d + tags + subtags + tag1_click_id + subtag1_click_id + \
           pctr_discrete_id + pctr + \
           user_tags_click + user_subtags_click + candidate_title + candidate_subtitle + \
           candidate_title_len + candidate_subtitle_len + \
           [click_title_list] + [click_subtitle_list] + [click_title_len_list] + \
           [click_subtitle_len_list] + label


def data_reader(dirname):
    """
    Data reader
    """

    def reader():
        """
        Reader
        """
        for fn in os.listdir(dirname):
            with open(dirname + "/" + fn, 'r') as f:
                for line in f:
                    # arr = line.strip().split(' ')
                    # arr = json.loads(line)
                    # res = del_sample_json(arr)
                    arr = line.strip().split('\t')
                    res = del_sample_same('\t'.join(arr[2:]))
                    yield res

    return reader


def model():
    """model"""
    user_phone_brand_id = layers.data(name='user_phone_brand', shape=[1], dtype='int64')
    user_gender_id = layers.data(name='user_gender', shape=[1], dtype='int64')
    user_age_id = layers.data(name='user_age', shape=[1], dtype='int64')
    user_status_id = layers.data(name='user_status', shape=[1], dtype="int64")
    user_trade_id = fluid.layers.data(name='user_trade', shape=[1], dtype='int64')
    user_cater_id = fluid.layers.data(name='user_cater', shape=[1], dtype='int64')
    user_income_id = fluid.layers.data(name='user_income', shape=[1], dtype='int64')

    user_city_id = fluid.layers.data(name='user_city', shape=[1], dtype='int64')

    user_click_id = fluid.layers.data(name='user_click', shape=[1], dtype='int64')
    user_b_click_id = fluid.layers.data(name='user_b_click', shape=[1], dtype='int64')
    user_c_click_id = fluid.layers.data(name='user_c_click', shape=[1], dtype='int64')
    user_d_click_id = fluid.layers.data(name='user_d_click', shape=[1], dtype='int64')

    week_id = layers.data(name='week', shape=[1], dtype="int64")
    hour_id = layers.data(name='hour', shape=[1], dtype='int64')

    content_b_c_d_id = layers.data(name='content_b_c_d', shape=[1], dtype='int64')
    content_tags_id = layers.data(name='content_tags', shape=[1], dtype='int64', lod_level=1)
    content_subtags_id = layers.data(name='content_subtags', shape=[1], dtype='int64', lod_level=1)

    user_content_tag_click_id = layers.data(name='user_content_tag_click', shape=[1], dtype='int64')
    user_content_subtag_click_id = layers.data(name='user_content_subtag_click', shape=[1], dtype='int64')

    content_pctr_discrete_id = layers.data(name='content_pctr_discrete', shape=[1], dtype='int64')
    # dnn_score_discrete_id = layers.data(name='dnn_score_discrete', shape=[1], dtype='int64')

    content_pctr = layers.data(name='content_pctr', shape=[1], dtype='float32')
    # dnn_score = layers.data(name='dnn_score', shape=[1], dtype='float32')
    # content_emb = layers.data(name='content_emb', shape=[64], dtype='float32')
    # user_emb = layers.data(name='user_emb', shape=[64], dtype='float32')

    user_click_tags_id = layers.data(
        name='user_click_tags_id', shape=[1], dtype='int64', lod_level=1)
    user_click_subtags_id = layers.data(
        name='user_click_subtags_id', shape=[1], dtype='int64', lod_level=1)
    candidate_title_word = layers.data(name='candidate_title', shape=[1], dtype='int64', lod_level=1)
    candidate_subtitle_word = layers.data(name='candidate_subtitle', shape=[1], dtype='int64', lod_level=1)
    candidate_title_len_id = layers.data(name='candidate_title_len', shape=[1], dtype='int64')
    candidate_subtitle_len_id = layers.data(name='candidate_subtitle_len', shape=[1], dtype='int64')

    click_title_list = layers.data(name='click_title_list', shape=[1], dtype='int64', lod_level=2)
    click_subtitle_list = layers.data(name='click_subtitle_list', shape=[1], dtype='int64', lod_level=2)
    click_title_len_list = layers.data(name='click_title_len_list', shape=[1], dtype='int64', lod_level=1)
    click_subtitle_len_list = layers.data(name='click_subtitle_len_list', shape=[1], dtype='int64', lod_level=1)

    label = layers.data(name='label', shape=[1], dtype='int64')
    # dnn_score_discrete_id.name, dnn_score.name, content_emb.name,user_emb.name,
    load_list = [user_phone_brand_id, user_gender_id, user_age_id,
                  user_status_id, user_trade_id, user_cater_id, user_income_id,
                  user_city_id, user_click_id, user_b_click_id, user_c_click_id,
                  user_d_click_id, week_id, hour_id, content_b_c_d_id,
                  content_tags_id, content_subtags_id, user_content_tag_click_id,
                  user_content_subtag_click_id, content_pctr_discrete_id,
                  content_pctr,
                  user_click_tags_id, user_click_subtags_id, candidate_title_word,
                  candidate_subtitle_word, candidate_title_len_id, candidate_subtitle_len_id,
                  click_title_list, click_subtitle_list,
                  click_title_len_list, click_subtitle_len_list,
                  label]
    feed_order = [x.name for x in load_list]

    user_phone_brand_emb = layers.embedding(
        input=user_phone_brand_id, dtype='float32',
        size=[7, EMB_LEN], param_attr='user_phone_brand_emb', is_sparse=True)
    user_gender_emb = layers.embedding(
        input=user_gender_id, dtype='float32',
        size=[3, EMB_LEN], param_attr='user_gender_emb', is_sparse=True)
    user_age_emb = layers.embedding(
        input=user_age_id, dtype='float32',
        size=[8, EMB_LEN], param_attr='user_age_emb', is_sparse=True)
    user_status_emb = layers.embedding(
        input=user_status_id, dtype='float32',
        size=[3, EMB_LEN], is_sparse=True, param_attr='user_status_emb')
    user_trade_emb = layers.embedding(
        input=user_trade_id, dtype='float32',
        size=[24, EMB_LEN], is_sparse=True, param_attr='user_trade_emb')
    user_cater_emb = layers.embedding(
        input=user_cater_id, dtype='float32',
        size=[4, EMB_LEN], is_sparse=True, param_attr='user_cater_emb')
    user_income_emb = layers.embedding(
        input=user_income_id, dtype='float32',
        size=[6, EMB_LEN], is_sparse=True, param_attr='user_income_emb')

    user_city_emb = layers.embedding(
        input=user_city_id, dtype='float32',
        size=[4000, EMB_LEN], is_sparse=True, param_attr='user_city_emb')

    user_click_emb = layers.embedding(
        input=user_click_id, dtype='float32',
        size=[6, EMB_LEN], is_sparse=True, param_attr='user_click_emb')
    user_b_click_emb = layers.embedding(
        input=user_b_click_id, dtype='float32',
        size=[6, EMB_LEN], is_sparse=True, param_attr='user_b_click_emb')
    user_c_click_emb = layers.embedding(
        input=user_c_click_id, dtype='float32',
        size=[6, EMB_LEN], is_sparse=True, param_attr='user_c_click_emb')
    user_d_click_emb = layers.embedding(
        input=user_d_click_id, dtype='float32',
        size=[6, EMB_LEN], is_sparse=True, param_attr='user_d_click_emb')

    week_emb = layers.embedding(
        input=week_id, dtype='float32',
        size=[8, EMB_LEN], is_sparse=True, param_attr='week_emb')
    hour_emb = layers.embedding(
        input=hour_id, dtype='float32',
        size=[24, EMB_LEN], is_sparse=True, param_attr='hour_emb')

    content_b_c_d_emb = layers.embedding(
        input=content_b_c_d_id, dtype='float32',
        size=[3, EMB_LEN], is_sparse=True, param_attr='content_b_c_d_emb')

    content_tags_emb = layers.embedding(
        input=content_tags_id, size=[11, EMB_LEN], dtype='float32', is_sparse=True,
        param_attr=fluid.ParamAttr(
            name="content_tags_emb", learning_rate=0.5, regularizer=fluid.regularizer.L2Decay(1.0))
    )
    content_tags_emb_avg = fluid.layers.sequence_pool(input=content_tags_emb, pool_type='average')

    content_subtags_emb = layers.embedding(
        input=content_subtags_id, size=[65, EMB_LEN], dtype='float32', is_sparse=True,
        param_attr=fluid.ParamAttr(
            name="content_subtags_emb", learning_rate=0.5,
            regularizer=fluid.regularizer.L2Decay(1.0))
    )
    content_subtags_emb_avg = fluid.layers.sequence_pool(
        input=content_subtags_emb, pool_type='average')

    user_content_tag_click_emb = layers.embedding(
        input=user_content_tag_click_id, dtype='float32',
        size=[11 * 6, EMB_LEN], is_sparse=True, param_attr='user_content_tag_click_emb')
    user_content_subtag_click_emb = layers.embedding(
        input=user_content_subtag_click_id, dtype='float32',
        size=[65 * 6, EMB_LEN], is_sparse=True, param_attr='user_content_subtag_click_emb')

    content_pctr_discrete_emb = layers.embedding(
        input=content_pctr_discrete_id, dtype='float32',
        size=[55, EMB_LEN], is_sparse=True, param_attr='content_pctr_discrete_emb')
    # dnn_score_discrete_emb = layers.embedding(
    #     input=dnn_score_discrete_id, dtype='float32',
    #     size=[21, EMB_LEN], is_sparse=True, param_attr='dnn_score_discrete_emb')

    user_click_tags_id_emb = layers.embedding(
        input=user_click_tags_id, size=[11 * 6, EMB_LEN], dtype='float32', is_sparse=True,
        param_attr="user_content_tag_click_emb")
    user_click_tags_id_emb_avg = fluid.layers.sequence_pool(
        input=user_click_tags_id_emb, pool_type='average')
    user_click_subtags_id_emb = layers.embedding(
        input=user_click_subtags_id, size=[65 * 6, EMB_LEN], dtype='float32', is_sparse=True,
        param_attr="user_content_subtag_click_emb")
    user_click_subtags_id_emb_avg = fluid.layers.sequence_pool(
        input=user_click_subtags_id_emb, pool_type='average')

    # 候选内容feature生成
    cand_title_emb = layers.embedding(input=candidate_title_word, size=[19962, EMB_LEN], dtype='float32',
                                      is_sparse=False, param_attr='word_embedding')
    cand_title_conv_pool = nets.sequence_conv_pool(
        input=cand_title_emb, num_filters=NUM_FILTERS, filter_size=3,
        act="relu", pool_type="average", param_attr='title_emb_conv', bias_attr='title_emb_conv_b')

    cand_subtitle_emb = layers.embedding(input=candidate_subtitle_word, size=[19962, EMB_LEN], dtype='float32',
                                         is_sparse=False, param_attr='word_embedding')
    cand_subtitle_conv_pool = nets.sequence_conv_pool(
        input=cand_subtitle_emb, num_filters=NUM_FILTERS, filter_size=3,
        act="relu", pool_type="average", param_attr='subtitle_emb_conv', bias_attr='subtitle_emb_conv_b')

    cand_title_len_emb = layers.embedding(input=candidate_title_len_id, size=[100, EMB_LEN], dtype='float32',
                                          is_sparse=True, param_attr='title_len_emb')
    cand_subtitle_len_emb = layers.embedding(input=candidate_subtitle_len_id, size=[100, EMB_LEN], dtype='float32',
                                             is_sparse=True, param_attr='subtitle_len_emb')

    cand_title_inf = layers.concat(
        input=[cand_title_conv_pool, cand_subtitle_conv_pool,
               cand_title_len_emb, cand_subtitle_len_emb], axis=-1)
    cand_title_feature = layers.fc(
        input=cand_title_inf, size=32, act="relu", param_attr='title_feature_list') #共享参数

    # 用户历史点击内容feature生成
    click_title_emb = layers.embedding(input=click_title_list, size=[19962, EMB_LEN], dtype='float32',
                                       is_sparse=False, param_attr='word_embedding')
    click_title_drnn = fluid.layers.DynamicRNN()
    with click_title_drnn.block():
        title_emb = click_title_drnn.step_input(click_title_emb)
        click_title_conv_pool = nets.sequence_conv_pool(
            input=title_emb, num_filters=NUM_FILTERS, filter_size=3,
            act="relu", pool_type="average", param_attr='title_emb_conv', bias_attr='title_emb_conv_b')
        click_title_drnn.output(click_title_conv_pool)
    click_title_conv_pool_list = click_title_drnn()

    click_subtitle_emb = layers.embedding(input=click_subtitle_list, size=[19962, EMB_LEN], dtype='float32',
                                       is_sparse=False, param_attr='word_embedding')
    click_subtitle_drnn = fluid.layers.DynamicRNN()
    with click_subtitle_drnn.block():
        subtitle_emb = click_subtitle_drnn.step_input(click_subtitle_emb)
        click_subtitle_conv_pool = nets.sequence_conv_pool(
            input=subtitle_emb, num_filters=NUM_FILTERS, filter_size=3,
            act="relu", pool_type="average", param_attr='subtitle_emb_conv', bias_attr='subtitle_emb_conv_b')
        click_subtitle_drnn.output(click_subtitle_conv_pool)
    click_subtitle_conv_pool_list = click_subtitle_drnn()

    click_title_len_emb_list = layers.embedding(input=click_title_len_list, size=[100, EMB_LEN], dtype='float32',
                                          is_sparse=True, param_attr='title_len_emb')
    click_subtitle_len_emb_list = layers.embedding(input=click_subtitle_len_list, size=[100, EMB_LEN], dtype='float32',
                                          is_sparse=True, param_attr='subtitle_len_emb')

    click_title_inf_list = layers.concat(
        input=[click_title_conv_pool_list, click_subtitle_conv_pool_list,
               click_title_len_emb_list, click_subtitle_len_emb_list], axis=-1)
    click_title_feature_list = layers.fc(
        input=click_title_inf_list, size=32, act="relu", param_attr='title_feature_list') #共享参数
    user_click_title_feature = layers.sequence_pool(input=click_title_feature_list, pool_type="average")

    user_emb_feature = layers.concat(
        input=[user_phone_brand_emb, user_gender_emb, user_age_emb, user_status_emb, user_trade_emb,
               user_cater_emb, user_income_emb, user_city_emb,
               user_click_emb, user_b_click_emb, user_c_click_emb, user_d_click_emb], axis=1)
    content_emb_feature = layers.concat(
        input=[content_b_c_d_emb, content_tags_emb_avg, content_subtags_emb_avg,
               content_pctr_discrete_emb, cand_title_feature], axis=1)
    cross_emb_feature = layers.concat(
        input=[user_content_tag_click_emb, user_content_subtag_click_emb,
               user_click_tags_id_emb_avg, user_click_subtags_id_emb_avg,
               user_click_title_feature], axis=1)
    env_emb_feature = layers.concat(
        input=[week_emb, hour_emb], axis=1)

    combined_features = layers.concat(input=[
        user_emb_feature, content_emb_feature, cross_emb_feature, env_emb_feature], axis=1)

    fc1 = layers.fc(input=combined_features, size=200, act='relu', param_attr='fc1', bias_attr='fc1_b')
    fc2 = layers.fc(input=fc1, size=200, act="relu", param_attr='fc2', bias_attr='fc2_b')
    fc3 = layers.fc(input=fc2, size=200, act="relu", param_attr='fc3', bias_attr='fc3_b')

    content_pctr_discrete_id_one_hot = layers.one_hot(
        content_pctr_discrete_id, 55, allow_out_of_range=False)

    final_layer = layers.concat(input=[fc3, content_pctr, content_pctr_discrete_id_one_hot], axis=1)
    predict = layers.fc(
        input=final_layer, size=2, act="softmax",
        param_attr='final_predict', bias_attr='final_predict_b')

    auc = fluid.layers.auc(
        input=predict, label=label, num_thresholds=2 ** 12)
    cost = layers.cross_entropy(input=predict, label=label)
    avg_cost = layers.reduce_mean(cost)

    loader = fluid.io.DataLoader.from_generator(
        feed_list=load_list, capacity=256, use_double_buffer=True, iterable=True)

    return {'predict': predict, 'avg_cost': avg_cost, 'feed_order': feed_order, 'loader': loader, 'auc': auc}


def infer(data_dir, model_dir, feed_list):
    """infer"""
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inference_scope = fluid.core.Scope()
    print "infer \n"
    test_reader = paddle.batch(
        data_reader(data_dir), batch_size=BATCH_SIZE)

    feeder = fluid.DataFeeder(feed_list, place)
    # exe.run(startup_program)
    exe = fluid.Executor(place)

    with fluid.scope_guard(inference_scope):
        startup_program = fluid.framework.Program()
        test_program = fluid.framework.Program()
        with fluid.framework.program_guard(test_program, startup_program):
            with fluid.unique_name.guard():
                model_args = model()
                auc_states = model_args['auc'][2]

        with open("infer_startup.proto", "w") as f:
            f.write(str(startup_program))
        [inference_program, _, fetch_targets] = fluid.io.load_inference_model(model_dir, exe)
        with open("inference_program.proto", "w") as f:
            f.write(str(inference_program))

        def set_zero(var_name):
            """
            zet zero
            """
            param = inference_scope.var(var_name).get_tensor()
            param_array = np.zeros(param._get_dims()).astype("int64")
            param.set(param_array, place)

        for auc_state in auc_states:
            set_zero(auc_state.name)

        cost_list = []
        auc_list = []
        for test_data in test_reader():
            _, cost, auc = exe.run(program=inference_program,
                                   feed=feeder.feed(test_data),
                                   fetch_list=fetch_targets)
            cost_list.append(np.array(cost))
            auc_list.append(np.array(auc))

        avg_cost = np.array(cost_list).mean()
        avg_auc = np.array(auc_list).mean()
        print "Test : cost %s ,auc %s" % (avg_cost, avg_auc)


def train(use_cuda, save_dirname, is_local, is_increment):
    """
    train
    """
    # predict, avg_cost, feed_order, auc_var, auc_batch, auc_states = model()
    old_model = None
    model_args = model()
    predict = model_args['predict']
    avg_cost = model_args['avg_cost']
    feed_order = model_args['feed_order']
    loader = model_args['loader']
    auc_batch = model_args['auc'][1]

    # 加入 fleet distributed_optimizer 加入分布式策略配置及多机优化
    sgd_optimizer = AdamOptimizer(learning_rate=2e-4)
    # sgd_optimizer = fluid.optimizer.Adam(learning_rate=2e-5)

    if is_local:
        sgd_optimizer.minimize(avg_cost)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

        exe = Executor(place)
        readers = []
        for i in range(16):
            readers.append(data_reader(cluster_train_dir))
        multi_readers = paddle.reader.multiprocess_reader(readers)
        loader.set_sample_generator(
            multi_readers, batch_size=BATCH_SIZE, places=fluid.cpu_places(CPU_NUM))
            # data_reader(cluster_train_dir), batch_size=BATCH_SIZE, places=fluid.cpu_places(CPU_NUM))
        # feeder = fluid.DataFeeder(feed_order, place)
        # train_reader = feeder.decorate_reader(
        #     paddle.batch(paddle.reader.shuffle(
        #         data_reader(cluster_train_dir), buf_size=8192), batch_size=BATCH_SIZE),
        #          multi_devices=False, drop_last=True)

        start_program = fluid.default_startup_program()
        exe.run(start_program)
        main_prog = fluid.default_main_program()

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = CPU_NUM * 2
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce # cpu reduce faster
        build_strategy.fuse_broadcast_ops = True
        # build_strategy.async_mode = True
        main_program = fluid.CompiledProgram(main_prog).with_data_parallel(
            loss_name=avg_cost.name, exec_strategy=exec_strategy, build_strategy=build_strategy)
            #loss_name=avg_cost.name, exec_strategy=exec_strategy, build_strategy=build_strategy, places=fluid.cpu_places(CPU_NUM))

        if is_increment:  # load model to fine-tune
            fluid.io.load_params(exe, old_model, main_program)
            for auc_state in model_args['auc'][2]:
                set_zero(place, fluid.global_scope(), auc_state.name)

        # 并行训练，速度更快
        # train_pe = fluid.ParallelExecutor(use_cuda=use_cuda,
        #                                   main_program=main_program, loss_name=avg_cost.name,
        #                                   exec_strategy=exec_strategy, build_strategy=build_strategy)

        cost_list = []
        auc_list = []
        import time
        pass_s_time = time.time()
        for pass_id in range(PASS_NUM):
            s_time = time.time()
            for batch_id, data in enumerate(loader()):
                r_time = time.time() - s_time
                st_time = time.time()
                cost_value, auc_value = exe.run(
                    program=main_program,
                    feed=data,
                    fetch_list=[avg_cost.name, auc_batch.name])
                t_time = time.time() - st_time
                cost_list.append(np.array(cost_value))
                auc_list.append(np.array(auc_value))

                if batch_id % 10 == 0 and batch_id != 0:
                    print "Pass %d, batch %d, cost %s auc %s readtime %f triantime %f" % \
                          (pass_id, batch_id, np.array(cost_list).mean(),
                           np.array(auc_list).mean(), r_time, t_time)
                    cost_list = []
                    auc_list = []
                if batch_id % 1000 == 0:
                    if save_dirname is not None:
                        fluid.io.save_inference_model(
                            save_dirname,
                            feed_order,
                            [predict, avg_cost, auc_batch], exe
                        )
                        fluid.io.save_persistables(exe, save_dirname)
                        infer(cluster_test_dir, save_dirname, feed_order)
                s_time = time.time()
        pass_time = time.time() - pass_s_time
        print("Pass train time: %f" % pass_time)

    else:
        role = role_maker.PaddleCloudRoleMaker()
        # 全异步训练
        config = DistributeTranspilerConfig()
        config.sync_mode = False
        config.runtime_split_send_recv = True
        # 加入 fleet init 初始化环境
        fleet.init(role)

        optimizer = fleet.distributed_optimizer(sgd_optimizer, config)
        optimizer.minimize(avg_cost)

        if fleet.is_server():
            fleet.init_server()
            fleet.run_server()
        # 启动worker
        if fleet.is_worker():
            # 初始化worker配置
            fleet.init_worker()

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = Executor(place)

            feeder = fluid.DataFeeder(feed_order, place)
            train_reader = feeder.decorate_reader(
                paddle.batch(paddle.reader.shuffle(
                    data_reader(cluster_train_dir), buf_size=8192), batch_size=BATCH_SIZE),
                multi_devices=False, drop_last=True)

            exe.run(fleet.startup_program)

            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_threads = CPU_NUM
            build_strategy = fluid.BuildStrategy()
            build_strategy.async_mode = True

            if CPU_NUM > 1:
                build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

            compiled_prog = fluid.compiler.CompiledProgram(
                fleet.main_program).with_data_parallel(
                loss_name=avg_cost.name, build_strategy=build_strategy, exec_strategy=exec_strategy)

            for pass_id in range(PASS_NUM):
                cost_list = []
                auc_list = []
                import time
                s_time = time.time()
                for batch_id, data in enumerate(train_reader()):
                    r_time = time.time() - s_time
                    cost_value, auc_value = exe.run(
                        program=compiled_prog, feed=data,
                        fetch_list=[avg_cost.name, auc_batch.name])
                    t_time = time.time() - r_time
                    cost_list.append(np.array(cost_value))
                    auc_list.append(np.array(auc_value))

                    if batch_id % 10 == 0 and batch_id != 0:
                        print "Pass %d, batch %d, cost %s auc %s readtime %f traintime %f" % \
                              (pass_id, batch_id, np.array(cost_list).mean(),
                               np.array(auc_list).mean(), r_time, t_time)
                        cost_list = []
                        auc_list = []
                    if batch_id % 1000 == 0 and fleet.is_first_worker():
                        if save_dirname is not None:
                            fleet.save_inference_model(
                                exe,
                                save_dirname,
                                feed_order,
                                [predict, avg_cost, auc_batch]
                            )
                            fleet.save_persistables(exe, save_dirname)
                            infer(cluster_test_dir, save_dirname, feed_order)
                    s_time = time.time()
        fleet.stop_worker()


def main(use_cuda):
    """
    main: train and infer
    """
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    # Directory for saving the inference model
    save_dirname = "./output/model/"
    if not os.path.isdir(save_dirname):
        os.makedirs(save_dirname)
    train(use_cuda, save_dirname, True, False)
    #  infer(use_cuda, save_dirname)


if __name__ == '__main__':
    use_cuda = os.getenv("PADDLE_USE_GPU", "0") == "1"
    main(use_cuda)

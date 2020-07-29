#!/bin/env python
# -*- encoding:utf-8 -*-
"""
activity uplift model
Authors: zhangtai
refer to:
"""
import math
import sys
import os
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

from paddle.fluid.executor import Executor
from paddle.fluid.optimizer import AdamOptimizer
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

USE_GPU = False
BATCH_SIZE = 1024
EMB_LEN = 16
PASS_NUM = 1
CPU_NUM = 4


def time_gap_discrete(time_gap):
    """
    time_gap discrete -> 7 id
    """
    time_gap_int = int(time_gap)
    if time_gap_int <= 1:
        discrete_res = 0
    elif time_gap_int <= 3:
        discrete_res = 1
    elif time_gap_int <= 5:
        discrete_res = 2
    elif time_gap_int <= 7:
        discrete_res = 3
    elif time_gap_int <= 14:
        discrete_res = 4
    elif time_gap_int <= 21:
        discrete_res = 5
    else:
        discrete_res = 6

    return discrete_res


def click_discrete(click_num):
    """
    count discrete -> 7 id
    """
    click_num_int = int(click_num)
    if click_num_int <= 0:
        discrete_res = 0
    elif click_num_int == 1:
        discrete_res = 1
    elif click_num_int <= 3:
        discrete_res = 2
    elif click_num_int <= 5:
        discrete_res = 3
    elif click_num_int <= 10:
        discrete_res = 4
    elif click_num_int <= 20:
        discrete_res = 5
    else:
        discrete_res = 6

    return discrete_res


def money_discrete(score):
    """
    score discrete -> 7 id
    """
    score_float = float(score)
    if score_float <= 0:
        discrete_res = 0
    elif score_float <= 1:
        discrete_res = 1
    elif score_float <= 3:
        discrete_res = 2
    elif score_float <= 5:
        discrete_res = 3
    elif score_float <= 10:
        discrete_res = 4
    elif score_float <= 20:
        discrete_res = 5
    else:
        discrete_res = 6

    return discrete_res


def del_sample(line):
    """
    deal input line to feature
    """
    arr = line.strip().split('\t')
    feature_and_label_arr = arr[1].split('#')
    features = feature_and_label_arr[0]
    labels = feature_and_label_arr[1].split(';')[0].split(',')

    feature_slots = [x.split(',') for x in features.split(';')]

    time_gap = [time_gap_discrete(feature_slots[0][0])]

    sex = [int(feature_slots[1][0])]
    age = [int(feature_slots[1][1])]
    car_owner = [int(feature_slots[1][2])]
    occupation = [int(feature_slots[1][3])]
    catering_expense_level = [int(feature_slots[1][4])]
    income_level = [int(feature_slots[1][5])]

    city_id = [int(feature_slots[2][0])]

    icon_click = [click_discrete(feature_slots[3][0])]
    banner_click = [click_discrete(feature_slots[3][1])]

    var_names = locals()
    for i in range(10):
        var_names["user_in_map_" + str(i)] = click_discrete(feature_slots[4][i])
    for i in range(10):
        var_names["user_in_navi_" + str(i)] = click_discrete(feature_slots[5][i])
    for i in range(10):
        var_names["user_in_voice_navi_" + str(i)] = click_discrete(feature_slots[6][i])
    for i in range(10):
        var_names["user_in_speed_navi_" + str(i)] = click_discrete(feature_slots[7][i])

    activity_uv = [click_discrete(feature_slots[8][0])]

    give_money_uv = [click_discrete(feature_slots[9][0])]
    give_money_discrete = [money_discrete(feature_slots[9][1])]

    get_money_uv = [click_discrete(feature_slots[10][0])]
    get_money_discrete = [money_discrete(feature_slots[10][1])]

    if_weekend = [int(feature_slots[11][0])]
    if_activity_in_icon = [int(feature_slots[12][0])]
    if_activity_in_banner = [int(feature_slots[12][1])]

    navi_label = [int(labels[0])]
    voice_navi_label = [int(labels[1])]
    speed_navi_label = [int(labels[2])]

    return time_gap + sex + age + car_owner + occupation + catering_expense_level + income_level + \
           city_id + icon_click + banner_click + \
           [var_names["user_in_map_" + str(x)] for x in range(10)] + \
           [var_names["user_in_navi_" + str(x)] for x in range(10)] + \
           [var_names["user_in_voice_navi_" + str(x)] for x in range(10)] + \
           [var_names["user_in_speed_navi_" + str(x)] for x in range(10)] + \
           activity_uv + give_money_uv + give_money_discrete + get_money_uv + get_money_discrete + \
           if_weekend + if_activity_in_icon + if_activity_in_banner + \
           navi_label + voice_navi_label + speed_navi_label


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
                    res = del_sample(line.strip())
                    yield res

    return reader


def streaming_data_reader():
    """
    streaming_data_reader
    """

    def reader():
        """
        Reader
        """
        for line in sys.stdin:
            res = del_sample(line.strip())
            yield res

    return reader


def model():
    """model"""
    user_time_gap_id = layers.data(name='user_time_gap', shape=[1], dtype='int64')

    user_gender_id = layers.data(name='user_gender', shape=[1], dtype='int64')
    user_age_id = layers.data(name='user_age', shape=[1], dtype='int64')
    user_status_id = layers.data(name='user_status', shape=[1], dtype="int64")
    user_trade_id = fluid.layers.data(name='user_trade', shape=[1], dtype='int64')
    user_cater_id = fluid.layers.data(name='user_cater', shape=[1], dtype='int64')
    user_income_id = fluid.layers.data(name='user_income', shape=[1], dtype='int64')

    user_city_id = fluid.layers.data(name='user_city', shape=[1], dtype='int64')

    icon_click_id = fluid.layers.data(name='icon_click', shape=[1], dtype='int64')
    banner_click_id = fluid.layers.data(name='banner_click', shape=[1], dtype='int64')

    navi_use_layers = {}
    for i in range(10):
        navi_use_layers["user_in_map_id_" + str(i)] = fluid.layers.data(
            name="user_in_map_id_" + str(i), shape=[1], dtype='int64')
    for i in range(10):
        navi_use_layers["user_in_navi_id_" + str(i)] = fluid.layers.data(
            name="user_in_navi_id_" + str(i), shape=[1], dtype='int64')
    for i in range(10):
        navi_use_layers["user_in_voice_navi_id_" + str(i)] = fluid.layers.data(
            name="user_in_voice_navi_id_" + str(i), shape=[1], dtype='int64')
    for i in range(10):
        navi_use_layers["user_in_speed_navi_id_" + str(i)] = fluid.layers.data(
            name="user_in_speed_navi_id_" + str(i), shape=[1], dtype='int64')

    activity_uv_id = fluid.layers.data(name='activity_uv', shape=[1], dtype='int64')

    give_money_uv_id = fluid.layers.data(name='give_money_uv', shape=[1], dtype='int64')
    give_money_discrete_id = fluid.layers.data(name='give_money_discrete', shape=[1], dtype='int64')

    get_money_uv_id = fluid.layers.data(name='get_money_uv', shape=[1], dtype='int64')
    get_money_discrete_id = fluid.layers.data(name='get_money_discrete', shape=[1], dtype='int64')

    if_weekend_id = fluid.layers.data(name='if_weekend', shape=[1], dtype='int64')
    if_activity_in_icon_id = fluid.layers.data(name='if_activity_in_icon', shape=[1], dtype='int64')
    if_activity_in_banner_id = \
        fluid.layers.data(name='if_activity_in_banner', shape=[1], dtype='int64')

    navi_label_id = layers.data(name='navi_label', shape=[1], dtype='int64')
    voice_navi_label_id = layers.data(name='voice_navi_label', shape=[1], dtype='int64')
    speed_navi_label_id = layers.data(name='speed_navi_label', shape=[1], dtype='int64')

    feed_order = [user_time_gap_id.name, user_gender_id.name, user_age_id.name,
                 user_status_id.name, user_trade_id.name, user_cater_id.name, user_income_id.name,
                 user_city_id.name, icon_click_id.name, banner_click_id.name] + \
                 [navi_use_layers["user_in_map_id_" + str(x)].name for x in range(10)] + \
                 [navi_use_layers["user_in_navi_id_" + str(x)].name for x in range(10)] + \
                 [navi_use_layers["user_in_voice_navi_id_" + str(x)].name for x in range(10)] + \
                 [navi_use_layers["user_in_speed_navi_id_" + str(x)].name for x in range(10)] + \
                 [activity_uv_id.name, give_money_uv_id.name, give_money_discrete_id.name,
                 get_money_uv_id.name, get_money_discrete_id.name,
                 if_weekend_id.name, if_activity_in_icon_id.name, if_activity_in_banner_id.name,
                 navi_label_id.name, voice_navi_label_id.name, speed_navi_label_id.name]

    user_time_gap_emb = layers.embedding(
        input=user_time_gap_id, dtype='float32',
        size=[7, EMB_LEN], param_attr='user_time_gap_emb', is_sparse=True)
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

    icon_click_emb = layers.embedding(
        input=icon_click_id, dtype='float32',
        size=[7, EMB_LEN], is_sparse=True, param_attr='user_click_emb')
    banner_click_emb = layers.embedding(
        input=banner_click_id, dtype='float32',
        size=[7, EMB_LEN], is_sparse=True, param_attr='user_click_emb')

    for i in range(10):
        navi_use_layers["user_in_map_emb_" + str(i)] = layers.embedding(
            input=navi_use_layers["user_in_map_id_" + str(i)], dtype='float32',
            size=[7, EMB_LEN], is_sparse=True, param_attr="user_in_map_emb_" + str(i))
    for i in range(10):
        navi_use_layers["user_in_navi_emb_" + str(i)] = layers.embedding(
            input=navi_use_layers["user_in_navi_id_" + str(i)], dtype='float32',
            size=[7, EMB_LEN], is_sparse=True, param_attr="user_in_navi_emb_" + str(i))
    for i in range(10):
        navi_use_layers["user_in_voice_navi_emb_" + str(i)] = layers.embedding(
            input=navi_use_layers["user_in_voice_navi_id_" + str(i)], dtype='float32',
            size=[7, EMB_LEN], is_sparse=True, param_attr="user_in_voice_navi_emb_" + str(i))
    for i in range(10):
        navi_use_layers["user_in_speed_navi_emb_" + str(i)] = layers.embedding(
            input=navi_use_layers["user_in_speed_navi_id_" + str(i)], dtype='float32',
            size=[7, EMB_LEN], is_sparse=True, param_attr="user_in_speed_navi_emb_" + str(i))

    activity_uv_emb = layers.embedding(
        input=activity_uv_id, dtype='float32',
        size=[7, EMB_LEN], is_sparse=True, param_attr='activity_uv_emb')

    give_money_uv_emb = layers.embedding(
        input=give_money_uv_id, dtype='float32',
        size=[7, EMB_LEN], is_sparse=True, param_attr='give_money_uv_emb')
    give_money_discrete_emb = layers.embedding(
        input=give_money_discrete_id, dtype='float32',
        size=[7, EMB_LEN], is_sparse=True, param_attr='give_money_discrete_emb')

    get_money_uv_emb = layers.embedding(
        input=get_money_uv_id, dtype='float32',
        size=[7, EMB_LEN], is_sparse=True, param_attr='get_money_uv_emb')
    get_money_discrete_emb = layers.embedding(
        input=get_money_discrete_id, dtype='float32',
        size=[7, EMB_LEN], is_sparse=True, param_attr='get_money_discrete_emb')

    if_weekend_emb = layers.embedding(
        input=if_weekend_id, dtype='float32',
        size=[2, EMB_LEN], is_sparse=True, param_attr='if_weekend_emb')
    if_activity_in_icon_emb = layers.embedding(
        input=if_activity_in_icon_id, dtype='float32',
        size=[2, EMB_LEN], is_sparse=True, param_attr='if_activity_in_icon_emb')
    if_activity_in_banner_id_emb = layers.embedding(
        input=if_activity_in_banner_id, dtype='float32',
        size=[2, EMB_LEN], is_sparse=True, param_attr='if_activity_in_banner_id_emb')

    concat_emb_feature = layers.concat(
        input=[user_time_gap_emb, user_gender_emb, user_age_emb, user_status_emb, user_trade_emb,
               user_cater_emb, user_income_emb, user_city_emb,
               icon_click_emb, banner_click_emb] + \
              [navi_use_layers["user_in_map_emb_" + str(x)] for x in range(10)] + \
              [navi_use_layers["user_in_navi_emb_" + str(x)] for x in range(10)] + \
              [navi_use_layers["user_in_voice_navi_emb_" + str(x)] for x in range(10)] + \
              [navi_use_layers["user_in_speed_navi_emb_" + str(x)] for x in range(10)] + \
              [activity_uv_emb, give_money_uv_emb, give_money_discrete_emb,
               get_money_uv_emb, get_money_discrete_emb,
               if_weekend_emb, if_activity_in_icon_emb, if_activity_in_banner_id_emb], axis=1)

    navi_fc1 = layers.fc(input=concat_emb_feature, size=100, act='relu')
    navi_fc2 = layers.fc(input=navi_fc1, size=100, act="relu")
    navi_predict = layers.fc(input=navi_fc2, size=2, act="softmax")

    voice_navi_fc1 = layers.fc(input=concat_emb_feature, size=100, act='relu')
    voice_navi_fc2 = layers.fc(input=voice_navi_fc1, size=100, act="relu")
    voice_navi_predict = layers.fc(input=voice_navi_fc2, size=2, act="softmax")

    speed_navi_fc1 = layers.fc(input=concat_emb_feature, size=100, act='relu')
    speed_navi_fc2 = layers.fc(input=speed_navi_fc1, size=100, act="relu")
    speed_navi_predict = layers.fc(input=speed_navi_fc2, size=2, act="softmax")

    navi_cost = layers.cross_entropy(input=navi_predict, label=navi_label_id) #todo
    voice_navi_cost = layers.cross_entropy(input=voice_navi_predict, label=voice_navi_label_id)
    speed_navi_cost = layers.cross_entropy(input=speed_navi_predict, label=speed_navi_label_id)

    two_cost = fluid.layers.elementwise_add(navi_cost, voice_navi_cost)
    all_cost = fluid.layers.elementwise_add(two_cost, speed_navi_cost)
    avg_cost = layers.reduce_mean(all_cost)

    return {'predict': [navi_predict, voice_navi_predict, speed_navi_predict],
            'avg_cost': avg_cost, 'feed_order': feed_order}


def set_zero(place, scope, var_name):
    """
    zet zero
    """
    param = scope.var(var_name).get_tensor()
    param_array = np.zeros(param._get_dims()).astype("int64")
    param.set(param_array, place)


def infer(data_dir, model_dir, feed_list):
    """infer"""
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inference_scope = fluid.Scope()
    print "infer \n"

    feeder = fluid.DataFeeder(feed_list, place)
    # exe.run(startup_program)
    exe = fluid.Executor(place)

    with fluid.scope_guard(inference_scope):
        startup_program = fluid.framework.Program()
        test_program = fluid.framework.Program()
        with fluid.program_guard(test_program, startup_program):
            with fluid.unique_name.guard():
                model_args = model()
                avg_cost = model_args['avg_cost']
                # auc_batch = model_args['auc'][1]
                # auc_states = model_args['auc'][2]

        fluid.io.load_persistables(executor=exe, dirname=model_dir, main_program=test_program)

        # for auc_state in auc_states:
        #     set_zero(place, inference_scope, auc_state.name)

        cost_list = []
        test_reader = paddle.batch(
            data_reader(data_dir), batch_size=BATCH_SIZE)
        for test_data in test_reader():
            cost = exe.run(program=test_program,
                           feed=feeder.feed(test_data),
                           fetch_list=[avg_cost])
            cost_list.append(np.array(cost))

        avg_cost = np.array(cost_list).mean()
        print "Test : cost %s" % (avg_cost,)


def train(use_cuda, train_sample_dir, test_sample_dir,
          old_model, output_model, is_local, is_increment):
    """
    train
    """
    # predict, avg_cost, feed_order, auc_var, auc_batch, auc_states = model()
    model_args = model()
    navi_predict = model_args['predict'][0]
    voice_navi_predict = model_args['predict'][1]
    speed_navi_predict = model_args['predict'][2]
    avg_cost = model_args['avg_cost']
    feed_order = model_args['feed_order']

    role = role_maker.PaddleCloudRoleMaker()
    # 全异步训练
    config = DistributeTranspilerConfig()
    config.sync_mode = False
    config.runtime_split_send_recv = True

    sgd_optimizer = AdamOptimizer(learning_rate=2e-4)

    if is_local:
        sgd_optimizer.minimize(avg_cost)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

        exe = Executor(place)
        # train_reader = paddle.batch(
        #     paddle.reader.shuffle(
        #         streaming_data_reader(), buf_size=8192), batch_size=BATCH_SIZE)

        feeder = fluid.DataFeeder(feed_order, place)
        train_reader = feeder.decorate_reader(
            paddle.batch(paddle.reader.shuffle(
                streaming_data_reader(), buf_size=8192), batch_size=BATCH_SIZE), 
            multi_devices=False, drop_last=True)
        start_program = fluid.default_startup_program()
        exe.run(start_program)
        main_program = fluid.default_main_program()
        if is_increment:  # load model to fine-tune
            fluid.io.load_params(exe, old_model, main_program)
            # for auc_state in model_args['auc'][2]:
            #     set_zero(place, fluid.global_scope(), auc_state.name)

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = CPU_NUM
        main_program.num_threads = CPU_NUM
        build_strategy = fluid.BuildStrategy()
        build_strategy.async_mode = True

        # 并行训练，速度更快
        train_pe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                          main_program=main_program, loss_name=avg_cost.name)

        cost_list = []
        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                cost_value = train_pe.run(
                    feed=data,
                    fetch_list=[avg_cost.name])
                cost_list.append(np.array(cost_value))

                if batch_id % 100 == 0 and batch_id != 0:
                    print "Pass %d, batch %d, cost %s" % \
                          (pass_id, batch_id, np.array(cost_list).mean())
                    cost_list = []
                if batch_id % 2000 == 0:
                    if output_model is not None:
                        fluid.io.save_inference_model(
                            output_model,
                            feed_order,
                            [navi_predict, voice_navi_predict, speed_navi_predict, avg_cost], exe
                        )
                        fluid.io.save_persistables(exe, output_model)
                        infer(test_sample_dir, output_model, feed_order)

    else:
        # 加入 fleet init 初始化环境
        fleet.init(role)
        # 加入 fleet distributed_optimizer 加入分布式策略配置及多机优化
        optimizer = fleet.distributed_optimizer(sgd_optimizer, config)
        optimizer.minimize(avg_cost)

        if fleet.is_server():
            if is_increment:
                fleet.init_server(old_model)
            else:
                fleet.init_server()
            fleet.run_server()
        # 启动worker
        if fleet.is_worker():
            # 初始化worker配置
            fleet.init_worker()

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

            exe = Executor(place)
            # train_reader = paddle.batch(
            #     paddle.reader.shuffle(
            #         data_reader(train_sample_dir), buf_size=8192), batch_size=BATCH_SIZE)

            feeder = fluid.DataFeeder(feed_order, place)
            train_reader = feeder.decorate_reader(
                paddle.batch(
                    paddle.reader.shuffle(
                        data_reader(train_sample_dir), buf_size=8192), 
                    batch_size=BATCH_SIZE), 
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

            cost_list = []
            for pass_id in range(PASS_NUM):
                for batch_id, data in enumerate(train_reader()):
                    cost_value = exe.run(
                        program=compiled_prog, feed=data,
                        fetch_list=[avg_cost.name])
                    cost_list.append(np.array(cost_value))

                    if batch_id % 100 == 0 and batch_id != 0:
                        print "Pass %d, batch %d, cost %s" % \
                              (pass_id, batch_id, np.array(cost_list).mean())
                        cost_list = []
                    if batch_id % 1000 == 0 and fleet.is_first_worker():
                        if output_model is not None:
                            fleet.save_inference_model(
                                exe,
                                output_model,
                                feed_order,
                                [navi_predict, voice_navi_predict, speed_navi_predict, avg_cost]
                            )
                            fleet.save_persistables(exe, output_model)
                            infer(test_sample_dir, output_model, feed_order)
        fleet.stop_worker()


def main(use_cuda, train_sample_dir=None, test_sample_dir=None, old_model=None, output_model=None):
    """
    main: train and infer
    """
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    # Directory for saving the inference model
    if not os.path.isdir(output_model):
        os.makedirs(output_model)
    train(use_cuda, train_sample_dir, test_sample_dir,
          old_model, output_model, False, False)
    #  infer(use_cuda, save_dirname)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'Usage: python_code train_sample_dir test_sample_dir old_model output_model'
        exit(1)

    use_cuda = os.getenv("PADDLE_USE_GPU", "0") == "1"
    main(use_cuda, sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

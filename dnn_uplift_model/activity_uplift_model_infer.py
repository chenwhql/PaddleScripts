# -*- coding: utf-8 -*-
"""
infer
"""
import os

import numpy as np
import paddle
import paddle.fluid as fluid
import sys
import json

# net_path = os.path.abspath(os.path.join('./'))
# sys.path.append(net_path)
import activity_uplift_model as uplift_dnn


def streaming_data_reader():
    """
    streaming_data_reader
    """

    def reader():
        """
        Reader
        """
        for line in sys.stdin:
            arr = line.strip().split('\t')
            cuid_prefix = arr[0].split('|')[0]
            feature_and_label_arr = arr[1].split('#')
            features = feature_and_label_arr[0]

            feature_slots = [x.split(',') for x in features.split(';')]

            time_gap = [uplift_dnn.time_gap_discrete(feature_slots[0][0])]

            sex = [int(feature_slots[1][0])]
            age = [int(feature_slots[1][1])]
            car_owner = [int(feature_slots[1][2])]
            occupation = [int(feature_slots[1][3])]
            catering_expense_level = [int(feature_slots[1][4])]
            income_level = [int(feature_slots[1][5])]

            city_id = [int(feature_slots[2][0])]

            icon_click = [uplift_dnn.click_discrete(feature_slots[3][0])]
            banner_click = [uplift_dnn.click_discrete(feature_slots[3][1])]

            var_names = locals()
            for i in range(10):
                var_names["user_in_map_" + str(i)] = uplift_dnn.click_discrete(feature_slots[4][i])
            for i in range(10):
                var_names["user_in_navi_" + str(i)] = uplift_dnn.click_discrete(feature_slots[5][i])
            for i in range(10):
                var_names["user_in_voice_navi_" + str(i)] = uplift_dnn.click_discrete(feature_slots[6][i])
            for i in range(10):
                var_names["user_in_speed_navi_" + str(i)] = uplift_dnn.click_discrete(feature_slots[7][i])

            activity_uv = [uplift_dnn.click_discrete(feature_slots[8][0])]

            give_money_uv = [uplift_dnn.click_discrete(feature_slots[9][0])]
            give_money_discrete = [uplift_dnn.money_discrete(feature_slots[9][1])]

            get_money_uv = [uplift_dnn.click_discrete(feature_slots[10][0])]
            get_money_discrete = [uplift_dnn.money_discrete(feature_slots[10][1])]

            if_weekend = [int(feature_slots[11][0])]

            for i in range(4):
                if i == 0:
                    if_activity_in_icon = [0]
                    if_activity_in_banner = [0]
                elif i == 1:
                    if_activity_in_icon = [1]
                    if_activity_in_banner = [0]
                elif i == 2:
                    if_activity_in_icon = [0]
                    if_activity_in_banner = [1]
                else:
                    if_activity_in_icon = [1]
                    if_activity_in_banner = [1]

                navi_label = [0]
                voice_navi_label = [0]
                speed_navi_label = [0]

                feature = time_gap + sex + age + car_owner + occupation + catering_expense_level + \
                    income_level + city_id + icon_click + banner_click + \
                    [var_names["user_in_map_" + str(x)] for x in range(10)] + \
                    [var_names["user_in_navi_" + str(x)] for x in range(10)] + \
                    [var_names["user_in_voice_navi_" + str(x)] for x in range(10)] + \
                    [var_names["user_in_speed_navi_" + str(x)] for x in range(10)] + \
                    activity_uv + give_money_uv + give_money_discrete + get_money_uv + get_money_discrete + \
                    if_weekend + if_activity_in_icon + if_activity_in_banner + \
                    navi_label + voice_navi_label + speed_navi_label

                yield (cuid_prefix, feature)

    return reader


def infer(test_reader, use_cuda, model_path=None):
    """
    inference function
    """
    if model_path is None:
        print(str(model_path) + " cannot be found")
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        startup_program = fluid.framework.Program()
        test_program = fluid.framework.Program()
        with fluid.framework.program_guard(test_program, startup_program):
            with fluid.unique_name.guard():
                model_args = uplift_dnn.model()
                navi_predict = model_args['predict'][0]
                voice_navi_predict = model_args['predict'][1]
                speed_navi_predict = model_args['predict'][2]
                feed_order = model_args['feed_order']

        fluid.io.load_persistables(executor=exe, dirname=model_path, main_program=test_program)

        feed_list = [
            test_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list, place)

        for data in test_reader():
            cuid_prefix_list = [x[0] for x in data]
            feature_list = [x[1] for x in data]

            navi_score, voice_navi_score, speed_navi_score = exe.run(
                test_program, feed=feeder.feed(feature_list), 
                fetch_list=[navi_predict, voice_navi_predict, speed_navi_predict], return_numpy=False)

            navi_score_list = np.array(navi_score).tolist()
            voice_navi_score_list = np.array(voice_navi_score).tolist()
            speed_navi_score_list = np.array(speed_navi_score).tolist()
            for i in range(len(navi_score_list) / 4):
                base_score = navi_score_list[i * 4][1] + \
                    voice_navi_score_list[i * 4][1] + speed_navi_score_list[i * 4][1]
                icon_score = navi_score_list[i * 4 + 1][1] + \
                    voice_navi_score_list[i * 4 + 1][1] + speed_navi_score_list[i * 4 + 1][1]
                banner_score = navi_score_list[i * 4 + 2][1] + \
                    voice_navi_score_list[i * 4 + 2][1] + speed_navi_score_list[i * 4 + 2][1]
                icon_banner_score = navi_score_list[i + 3][1] + \
                    voice_navi_score_list[i * 4 + 3][1] + speed_navi_score_list[i * 4 + 3][1]
                print "%s\t%f\t%f\t%f\t%f" % (
                    cuid_prefix_list[i * 4], base_score, icon_score, banner_score, icon_banner_score)


if __name__ == "__main__":

    if len(sys.argv) == 2:
        model_path = sys.argv[1]

        # print content_feature_list
        reader = streaming_data_reader()
        test_reader = paddle.batch(reader, batch_size=4 * 4)

        infer(test_reader, use_cuda=False, model_path=model_path)

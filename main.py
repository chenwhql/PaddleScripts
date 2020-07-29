#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import json
import paddle.fluid as fluid
import paddle
import numpy as np
import os
import getopt
import sys

conf = {
    'model_name': '',
    'model_file_name': '',
    'params_file_name': '',
    'isFromHub': False
}

def main():
    print(paddle.__version__)
    exe = fluid.Executor(fluid.CPUPlace())
    model_dir = "./model"
    model_name="huangfan"
    model_path = os.path.join(model_dir, model_name)
    model_filename = "model"
    params_filename = "params"
    [prog, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=model_path, executor=exe, model_filename=model_filename, params_filename=params_filename)

    data = np.random.rand(3,224,224)
    data = data.reshape(1, 3, 224, 224)
    data = data.astype("float32")

    for v in prog.list_vars():
        v.persistable = True

    results = exe.run(prog, feed={feed_target_names[0]: data}, fetch_list=fetch_targets)[0].tolist()[0]

    info = {"vars": [], "ops": []}

    # todo 方式一
    fetched_var = fluid.executor._fetch_var('batch_norm_36.tmp_0')
    print(fetched_var)

    # 获取program中所有的变量

    for v in prog.list_vars():
        fetched_var = fluid.executor._fetch_var(v.name)
        print(fetched_var)
    
    """
    for v in prog.list_vars():
        if not v.persistable:
            print(v.name)

        try:
            # todo 方式二
            new_fetch_targets = []
            new_fetch_targets.append(fetch_targets[0])
            new_fetch_targets.append(v)
            print(fetch_targets)
            results = exe.run(prog,fetch_list=new_fetch_targets,feed={feed_target_names[0]: data})

        except Exception as e:
            print('fail: ')
            print(e)
            print(v.name)
            continue
        else:
            print('success: ')
            print(v.name)
            data = results[0]
            v_info = {}
            v_info["name"] = v.name
            v_info["shape"] = list(data.shape)
            v_info["data"] = data.flatten().tolist()
            v_info["persistable"] = v.persistable
            info["vars"].append(v_info)
            print(info)

        break
    """


main()
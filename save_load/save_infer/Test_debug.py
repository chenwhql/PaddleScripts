# coding=utf-8
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# Disable Warning
import logging
import re
import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDNN_LOGINFO_DBG"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

logging.captureWarnings(True)
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}.'.format(re.escape(__name__)))
warnings.warn("This is a DeprecationWarning",category=DeprecationWarning)


import paddle
from paddle.io import IterableDataset, DataLoader
from paddle.static import InputSpec
# import paddlecloud.visual_util as visualdl
import numpy as np
import args
import os
import sys
import logging
import time
import random
from scipy.stats import pearsonr

from TestRankNet import RankNet as net

def set_seed(seed=2022):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def run_train():
    model = net(250)
    print("------------train model summary-----------")
    print(paddle.summary(model, [(-1,76), (-1,250,26)]))
    print("------------test state dict--------------")
    paddle.save(model.state_dict(), "model_state.pdparams")

def run_infer():
    new_model = net(250)
    state_dict = paddle.load("model_state.pdparams")
    new_model.set_state_dict(state_dict, use_structured_name=True)
    print(paddle.summary(new_model, [(-1,76), (-1,250,26)]))

if __name__ == "__main__":
    set_seed(seed=2022)
    run_train()
    run_infer()
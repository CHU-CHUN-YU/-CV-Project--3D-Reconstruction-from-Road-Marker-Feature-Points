import os
import sys
import math
from PIL import Image
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--result_name', type=str, default='TEST_ST_0')
opt = parser.parse_args()
print(opt)

camera_dict = pickle_load('./pkl/camera_dict.pkl')
train_sub_map = pickle_load('./pkl/sub_map.pkl')
train_init_pose = pickle_load('./pkl/new_train_init.pkl')
train_re = pickle_load(f'./result/{opt.result_name}/aug_point_cloud.pkl')

test_sub_map = pickle_load('./pkl/test_sub_map.pkl')
test_init_pose = pickle_load('./pkl/new_test_init.pkl')
test_re = pickle_load(f'./result/{opt.result_name}/aug_point_cloud.pkl')


result_name = opt.result_name
if opt.mode == 'train':
    re = generate_pred_pose(result_name, train_re, train_sub_map, train_init_pose, 0.2)
elif opt.mode == 'test':
    re = generate_pred_pose(result_name, test_re, test_sub_map, test_init_pose, 0.2, mode='test')
print(re)
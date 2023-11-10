import os
import sys
import math
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import open3d as o3d
import csv
import matplotlib.pyplot as plt

FR_to_F_Q = np.array([-0.0806252, 0.607127, 0.0356452, 0.789699])
FL_to_F_Q = np.array([-0.117199, -0.575476, -0.0686302, 0.806462])
B_to_FL_Q = np.array([0.074732, -0.794, -0.10595, 0.59393])
F_to_BASE_Q = np.array([-0.5070558775462676, 0.47615311808704197, -0.4812773544166568, 0.5334272708696808])

FR_to_F_T = np.array([0.559084, 0.0287952, -0.0950537])
FL_to_F_T = np.array([-0.564697, 0.0402756, -0.028059])
B_to_FL_T = np.array([-1.2446, 0.21365, -0.91917])
F_to_BASE_T = np.array([0.0, 0.0, 0.0])

P_B  = np.array([[547.741, 0.0,     715.998, 0.0], 
                 [0.0,     548.549, 478.83,  0.0], 
                 [0.0,     0.0,     1.0,     0.0]])
P_F  = np.array([[539.36,  0.0,     721.262, 0.0], 
                 [0.0,     540.02,  464.54,  0.0], 
                 [0.0,     0.0,     1.0,     0.0]])
P_FR = np.array([[542.867, 0.0,     739.613, 0.0],
                 [0.0,     542.872, 474.175, 0.0],
                 [0.0,     0.0,     1.0,     0.0]])
P_FL = np.array([[549.959, 0.0,     728.516, 0.0], 
                 [0.0,     549.851, 448.147, 0.0], 
                 [0.0,     0.0,     1.0,     0.0]])

D_B = np.array([-0.00790509510948, -0.0356504181626, 0.00803540169827, 0.0059685787996])
D_F = np.array([-0.0309599861474, 0.0195100168293, -0.0454086703952, 0.0244895806953])
D_FR = np.array([-0.0273485687967, 0.0202959209357, -0.0499610225624, 0.0261513487397])
D_FL = np.array([-0.0040468506737, -0.0433305077484, 0.0204357876847, -0.00127388969373])

K_B = np.array([[658.897676983, 0.0,           719.335668486], 
                [0.0,           659.869992391, 468.32106087], 
                [0.0,           0.0,           1.0]])
K_F = np.array([[661.949026684, 0.0,           720.264314891], 
                [0.0,           662.758817961, 464.188882538], 
                [0.0,           0.0,           1.0]])
K_FL = np.array([[658.929184246, 0.0,           721.005287695], 
                 [0.0,           658.798994733, 460.495402628], 
                 [0.0,           0.0,           1.0]])
K_FR = np.array([[660.195664468, 0.0,           724.021995966], 
                 [0.0,           660.202323944, 467.498636505], 
                 [0.0,           0.0,           1.0]])

def txt_load(DIR):
    '''
    :param DIR: {string} directory of input txt file
    :return: {list} list of all lines
    '''
    with open(DIR, 'r') as f:
        L = f.readlines()
    for i in range(len(L)):
        L[i] = L[i][:-1]
    return L

def pickle_save(D, filename):
    '''Save D into a pkl file
    :param D: {} saved variable
    :param filename: {string} directory and filename of saved dict
    '''
    with open(filename, 'wb') as f:
        pickle.dump(D, f)
        
def pickle_load(filename):
    '''load D pkl file
    :param filename: {string} directory and filename of laoded dict
    :return: {} loaded variable
    '''
    with open(filename, 'rb') as f:
        re = pickle.load(f)
    return re

def get_camera_dict(which='train'):
    '''
    :param which: {str} train or test
    :return: {dict} get camera name dict of all sequence and time stamp
    '''
    if which == 'train':
        camera_dict = {}
        for seq in ['seq1', 'seq2', 'seq3']:
            camera_dict[seq] = {}
            for time in os.listdir(f'./ITRI_dataset/{seq}/dataset'):
                camera_dict[seq][time] = pd.read_csv(f'./ITRI_dataset/{seq}/dataset/{time}/camera.csv').columns[0]
    elif which == 'test':
        camera_dict = {}
        for seq in ['test1', 'test2']:
            camera_dict[seq] = {}
            for time in os.listdir(f'./ITRI_DLC/{seq}/dataset'):
                camera_dict[seq][time] = pd.read_csv(f'./ITRI_DLC/{seq}/dataset/{time}/camera.csv').columns[0]
    return camera_dict

def get_road_marker_dict():
    '''
    :return: {dict} get road marker dict of all sequence and time stamp
    '''
    road_marker = {}
    for seq in ['seq1', 'seq2', 'seq3']:
        road_marker[seq] = {}
        for time in os.listdir(f'./ITRI_dataset/{seq}/dataset'):
            try:
                file = pd.read_csv(f'./ITRI_dataset/{seq}/dataset/{time}/detect_road_marker.csv')
            except:
                road_marker[seq][time] = -1
                continue
            bb = np.concatenate([np.array(file.columns).astype(np.float64)[np.newaxis, :], file.to_numpy()], axis=0)
            road_marker[seq][time] = bb
            del file, bb
    return road_marker

def plot_bounding_box(img, bounding_box):
    '''Function to print info on original image
    :param img: {numpy.ndarray} image array (IN RGB form)
    :param bounding_box: {numpy.ndarray} info of each bounding box (p1_w, p1_h, p2_w, p2_h, class, probability)
    :return: {numpy.ndarray} image with bounding box and text info
    '''
    cla_to_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    img_mod = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for l in range(len(bounding_box)):
        p1 = tuple(bounding_box[l][:2][::-1].astype(np.int32))
        p2 = tuple(bounding_box[l][2:4][::-1].astype(np.int32))
        cla = int(bounding_box[l][4])
        p = bounding_box[l][5]
        name = 'C = ' + str(cla) + ',P = ' + str(p)
        img_mod = cv2.rectangle(img_mod, (p1[1], p1[0]), (p2[1], p2[0]), cla_to_color[cla], 2)
        cv2.putText(img_mod, name, (p1[1], p1[0]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cla_to_color[cla], 1)
    img_mod = cv2.cvtColor(img_mod, cv2.COLOR_BGR2RGB)
    return img_mod

def get_initial_pose_dict():
    '''
    :return: {dict} get initial pose dict of all sequence and time stamp
    '''
    initial_pose = {}
    for seq in ['seq1', 'seq2', 'seq3']:
        initial_pose[seq] = {}
        for time in os.listdir(f'./ITRI_dataset/{seq}/dataset'):
            try:
                file = pd.read_csv(f'./ITRI_dataset/{seq}/dataset/{time}/initial_pose.csv')
            except:
                initial_pose[seq][time] = -1
                continue
            m = np.concatenate([np.array(file.columns).astype(np.float64)[np.newaxis, :], file.to_numpy()], axis=0)
            initial_pose[seq][time] = m
            del file, m
    return initial_pose

def get_new_initial_pose_dict():
    '''
    :return: {dict} get initial pose dict of all sequence and time stamp
    '''
    initial_pose = {}
    for seq in ['seq1', 'seq2', 'seq3']:
        initial_pose[seq] = {}
        for time in os.listdir(f'./ITRI_DLC2/{seq}/new_init_pose'):
            try:
                file = pd.read_csv(f'./ITRI_DLC2/{seq}/new_init_pose/{time}/initial_pose.csv')
            except:
                initial_pose[seq][time] = -1
                continue
            m = np.concatenate([np.array(file.columns).astype(np.float64)[np.newaxis, :], file.to_numpy()], axis=0)
            initial_pose[seq][time] = m
            del file, m
    return initial_pose

def get_new_test_initial_pose_dict():
    '''
    :return: {dict} get initial pose dict of all sequence and time stamp
    '''
    initial_pose = {}
    for seq in ['test1', 'test2']:
        initial_pose[seq] = {}
        for time in os.listdir(f'./ITRI_DLC2/{seq}/new_init_pose'):
            try:
                file = pd.read_csv(f'./ITRI_DLC2/{seq}/new_init_pose/{time}/initial_pose.csv')
            except:
                initial_pose[seq][time] = -1
                continue
            m = np.concatenate([np.array(file.columns).astype(np.float64)[np.newaxis, :], file.to_numpy()], axis=0)
            initial_pose[seq][time] = m
            del file, m
    return initial_pose

def get_gt():
    '''Function to get dict of ground truth
    :return: {dict} ground truth dict (missing value filled by -1)
    '''
    GT = {}
    for seq in ['seq1', 'seq2', 'seq3']:
        GT[seq] = {}
        for time in os.listdir(f'./ITRI_dataset/{seq}/dataset'):
            try:
                file = pd.read_csv(f'./ITRI_dataset/{seq}/dataset/{time}/gound_turth_pose.csv')
            except:
                GT[seq][time] = -1
                continue
            m = np.concatenate([np.array(file.columns).astype(np.float64)[np.newaxis, :], file.to_numpy()], axis=0)
            GT[seq][time] = m
            del file, m
    return GT

def get_sub_map():
    '''Function to get dict of sub_map
    :return: {dict} sub_map dict (missing value filled by -1)
    '''
    sub_map = {}
    for seq in ['seq1', 'seq2', 'seq3']:
        sub_map[seq] = {}
        for time in os.listdir(f'./ITRI_dataset/{seq}/dataset'):
            try:
                file = pd.read_csv(f'./ITRI_dataset/{seq}/dataset/{time}/sub_map.csv')
            except:
                sub_map[seq][time] = -1
                continue
            m = np.concatenate([np.array(file.columns).astype(np.float64)[np.newaxis, :], file.to_numpy()], axis=0)
            sub_map[seq][time] = m
            del file, m
    return sub_map

def get_test_sub_map():
    '''Function to get dict of sub_map
    :return: {dict} sub_map dict (missing value filled by -1)
    '''
    sub_map = {}
    for seq in ['test1', 'test2']:
        sub_map[seq] = {}
        for time in os.listdir(f'./ITRI_DLC/{seq}/dataset'):
            try:
                file = pd.read_csv(f'./ITRI_DLC/{seq}/dataset/{time}/sub_map.csv')
            except:
                sub_map[seq][time] = -1
                continue
            m = np.concatenate([np.array(file.columns).astype(np.float64)[np.newaxis, :], file.to_numpy()], axis=0)
            sub_map[seq][time] = m
            del file, m
    return sub_map

def quart_to_rpy(quaternion):
    '''
    :param quaternion: {tuple} Quaternion (x, y, z, w)
    :return: {tuple} 
    '''
    x, y, z, w = quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def eulerAnglesToRotationMatrix(theta):
    '''Function to convert Euler Angles to rotation matrix
    :param theta: {tuple, list, numpy.ndarray} (roll, pitch, yaw)
    :return: {numpy.ndarray} rotation matrix
    '''
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])            
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def transform_1coor(ori_coor, TYPE, show_front=False):
    '''
    :param ori_coor: {numpy.ndarray} original coordinate vector (x, y, z)
    :param TYPE: {string} transformation type ('fr2base' or 'fl2base' or 'f2base')
    :return: {numpy.ndarray} result coordinate vector (x', y', z')
    '''
    global FR_to_F_Q, FL_to_F_Q, FR_to_F_T, FL_to_F_T, F_to_BASE_Q, F_to_BASE_T
    if TYPE == 'fr2base':
        dst_coor = np.matmul(eulerAnglesToRotationMatrix(quart_to_rpy(FR_to_F_Q)), ori_coor)
        dst_coor = dst_coor + FR_to_F_T
    elif TYPE == 'fl2base':
        dst_coor = np.matmul(eulerAnglesToRotationMatrix(quart_to_rpy(FL_to_F_Q)), ori_coor)
        dst_coor = dst_coor + FL_to_F_T
    elif TYPE == 'f2base':
        dst_coor = ori_coor
    elif TYPE == 'b2base':
        dst_coor = np.matmul(eulerAnglesToRotationMatrix(quart_to_rpy(B_to_FL_Q)), ori_coor)
        dst_coor = dst_coor + B_to_FL_T
        dst_coor = np.matmul(eulerAnglesToRotationMatrix(quart_to_rpy(FL_to_F_Q)), dst_coor)
        dst_coor = dst_coor + FL_to_F_T
    if show_front:
        # print('Output f-camera coordinate!')
        return dst_coor
    dst_coor = np.matmul(eulerAnglesToRotationMatrix(quart_to_rpy(F_to_BASE_Q)), dst_coor)
    dst_coor = dst_coor + F_to_BASE_T
    return dst_coor

def transform_F_to_BASE(ori_coor):
    '''
    :param ori_coor: {numpy.ndarray} f camera coordinate vector (x, y, z)
    :return: {numpy.ndarray} result coordinate vector (x', y', z')
    '''
    global F_to_BASE_Q, F_to_BASE_T
    dst_coor = np.matmul(eulerAnglesToRotationMatrix(quart_to_rpy(F_to_BASE_Q)), ori_coor)
    dst_coor = dst_coor + F_to_BASE_T
    return dst_coor

def inv_transform_1coor(dst_coor, TYPE):
    '''
    :param dst_coor: {numpy.ndarray} original coordinate vector (x', y', z')
    :param TYPE: {string} transformation type ('base2f' or 'base2fr' or 'base2fl' or 'base2b')
    :return: {numpy.ndarray} result coordinate vector (x, y, z)
    '''
    global FR_to_F_Q, FL_to_F_Q, B_to_FL_Q, FR_to_F_T, FL_to_F_T, B_to_FL_T, F_to_BASE_Q, F_to_BASE_T
    ori_coor = dst_coor - F_to_BASE_T
    R_b2f = np.linalg.inv( eulerAnglesToRotationMatrix( quart_to_rpy( F_to_BASE_Q ) ) )
    ori_coor = np.matmul(R_b2f, ori_coor)
    if TYPE == 'base2fr':
        ori_coor = ori_coor - FR_to_F_T
        R_f2fr = np.linalg.inv( eulerAnglesToRotationMatrix( quart_to_rpy( FR_to_F_Q ) ) )
        ori_coor = np.matmul(R_f2fr, ori_coor)
    elif TYPE == 'base2fl':
        ori_coor = ori_coor - FL_to_F_T
        R_f2fr = np.linalg.inv( eulerAnglesToRotationMatrix( quart_to_rpy( FL_to_F_Q ) ) )
        ori_coor = np.matmul(R_f2fr, ori_coor)
    elif TYPE == 'base2f':
        pass
    elif TYPE == 'base2b':
        ori_coor = ori_coor - FL_to_F_T
        R_f2fr = np.linalg.inv( eulerAnglesToRotationMatrix( quart_to_rpy( FL_to_F_Q ) ) )
        ori_coor = np.matmul(R_f2fr, ori_coor)
        ori_coor = ori_coor - B_to_FL_T
        R_b2fl = np.linalg.inv( eulerAnglesToRotationMatrix( quart_to_rpy( B_to_FL_Q ) ) )
        ori_coor = np.matmul(R_b2fl, ori_coor)
    return ori_coor

def sub_map_recon(seq, time, filename=None):
    '''
    :param seq: {string} sequence name
    :param time: {string} timestamp
    :param filename: {string} file name to save result
    :return: {numpy.ndarray}
    '''
    camera_dict = pickle_load('./camera_dict.pkl')
    init_pose = pickle_load('./abandon/initial_pose.pkl')
    init_map = pickle_load('./sub_map.pkl')
    
    # Load R, T mtrix of init to base
    init_pc = init_map[seq][time]
    base_to_init_R = init_pose[seq][time][:3, :3]
    base_to_init_T = init_pose[seq][time][:3, 3]
    
    # convert sub map point cloud form init to base
    base_pc = init_pc.copy()
    for i in range(base_pc.shape[0]):
        base_pc[i] = np.matmul(np.linalg.inv(base_to_init_R), (init_pc[i] - base_to_init_T))
    
    # convert sub map point cloud form base to camera
    c_pc = base_pc.copy()
    for i in range(c_pc.shape[0]):
        if camera_dict[seq][time].split('_')[-2] == 'f':
            c_pc[i] = inv_transform_1coor(c_pc[i], 'base2f')
        if camera_dict[seq][time].split('_')[-2] == 'fr':
            c_pc[i] = inv_transform_1coor(c_pc[i], 'base2fr')
        if camera_dict[seq][time].split('_')[-2] == 'fl':
            c_pc[i] = inv_transform_1coor(c_pc[i], 'base2fl')
        if camera_dict[seq][time].split('_')[-2] == 'b':
            c_pc[i] = inv_transform_1coor(c_pc[i], 'base2b')
    if camera_dict[seq][time].split('_')[-2] == 'f':
        c_p = np.matmul(P_F[:, :3], c_pc.T).T        
    if camera_dict[seq][time].split('_')[-2] == 'fr':
        c_p = np.matmul(P_FR[:, :3], c_pc.T).T
    if camera_dict[seq][time].split('_')[-2] == 'fl':
        c_p = np.matmul(P_FL[:, :3], c_pc.T).T
    if camera_dict[seq][time].split('_')[-2] == 'b':
        c_p = np.matmul(P_B[:, :3], c_pc.T).T 
    c_p[:, 0] = c_p[:, 0]/c_p[:, 2]
    c_p[:, 1] = c_p[:, 1]/c_p[:, 2]

    # Load original image
    img = np.array(  Image.open(f'./ITRI_dataset/{seq}/dataset/{time}/raw_image.jpg')  )
    # only valid pixel coor need to be plot
    X = c_p[:, 0]
    Y = c_p[:, 1]
    mask = np.logical_and(np.logical_and(X >= 0, X < img.shape[0]), np.logical_and(Y >= 0, Y < img.shape[1]))
    X = X[mask].astype(np.int64)
    Y = Y[mask].astype(np.int64)
    
    for i in range(len(X)):
        cv2.circle(img, (X[i], Y[i]), radius=1, color=(255, 0, 0), thickness=2)
    if filename is not None:
        Image.fromarray(img).save(filename)
    return img, c_p[mask][:, :2]

def pixel_to_world(pixel_array, TYPE):
    '''
    :param pixel_array: {numpy.ndarray} array of all corner point (in pixel coor) with shape (N, 2)
    :param TYPE: {string} what camera input array belong to ('f' or 'fr' or 'fl' or 'b')
    :return: {numpy.ndarray} array of world point cloud (in original camera coor) with shape (N, 3)
    '''
    global P_B, P_F, P_FR, P_FL
    pixel_array = np.concatenate([pixel_array, np.ones(pixel_array.shape[0]).T[:, np.newaxis]], axis=1)
    if TYPE == 'f':
        point_cloud = np.matmul(np.linalg.inv(P_F[:, :3]), pixel_array.T).T
    elif TYPE == 'fr':
        point_cloud = np.matmul(np.linalg.inv(P_FR[:, :3]), pixel_array.T).T
    elif TYPE == 'fl':
        point_cloud = np.matmul(np.linalg.inv(P_FL[:, :3]), pixel_array.T).T
    elif TYPE == 'b':
        point_cloud = np.matmul(np.linalg.inv(P_B[:, :3]), pixel_array.T).T
    return point_cloud

def world_coor_transform(camera_point_cloud, TYPE, show_front=False):
    '''
    :param camera_point_cloud: {numpy.ndarray} array of point cloud using (in camera coor) with shape (N, 3)
    :param TYPE: {string} transformation type ('fr2base' or 'fl2base' or 'f2base' or 'b2base')
    '''
    for i in range(camera_point_cloud.shape[0]):
        camera_point_cloud[i] = transform_1coor(camera_point_cloud[i], TYPE, show_front)
    return camera_point_cloud

def world_coor_transform_high_eff(c_pc, TYPE):
    '''
    :param c_pc: {numpy.ndarray} array of point cloud using (in camera coor) with shape (N, 3)
    :param TYPE: {string} transformation type ('fr2base' or 'fl2base' or 'f2base' or 'b2base')
    '''
    global FR_to_F_Q, FL_to_F_Q, FR_to_F_T, FL_to_F_T, F_to_BASE_Q, F_to_BASE_T
    if TYPE == 'fr2base':
        c_pc = np.matmul(eulerAnglesToRotationMatrix(quart_to_rpy(FR_to_F_Q)), c_pc.T).T + FR_to_F_T
    elif TYPE == 'fl2base':
        c_pc = np.matmul(eulerAnglesToRotationMatrix(quart_to_rpy(FL_to_F_Q)), c_pc.T).T + FL_to_F_T
    elif TYPE == 'f2base':
        pass
    elif TYPE == 'b2base':
        c_pc = np.matmul(eulerAnglesToRotationMatrix(quart_to_rpy(B_to_FL_Q)), c_pc.T).T + B_to_FL_T
        c_pc = np.matmul(eulerAnglesToRotationMatrix(quart_to_rpy(FL_to_F_Q)), c_pc.T).T + FL_to_F_T
    c_pc = np.matmul(eulerAnglesToRotationMatrix(quart_to_rpy(F_to_BASE_Q)), c_pc.T).T + F_to_BASE_T
    return c_pc

def check_transform(ori_coor):
    '''Function to check the validity of forward/backward transformation (better check again by sub_map_recon())
    :param ori_coor: {numpy.ndarray} 3d vector (x, y, z)
    '''
    FORWARD = ['f2base', 'fr2base', 'fl2base', 'b2base']
    BACKEARD = ['base2f', 'base2fr', 'base2fl', 'base2b']
    for i in range(len(FORWARD)):
        wb = transform_1coor(ori_coor, 'f2base')
        w_rec = inv_transform_1coor(wb, 'base2f')
        print(np.isclose(ori_coor, w_rec).all())

###########################################################################################################

def get_bounding_box_region(img, bounding_box):
    '''Function to get dict of all bounding region
    :param img: {numpy.ndarray} image array (IN RGB form)
    :param bounding_box: {numpy.ndarray} info of each bounding box
    :return: {tuple} list of bounding region image and corresponding info
    '''
    patch_L = []
    info_L = []
    for l in range(len(bounding_box)):
        info = bounding_box[l]
        info[:2] = np.clip(np.floor(info[:2]), 0, max(img.shape[:2]))
        info[2:4] = np.clip(np.ceil(info[2:4]), 0, max(img.shape[:2]))
        info = info[:5].astype(np.int64)
        info_L.append( info )
        patch_L.append( img[info[1]:info[3], info[0]:info[2], :] )
    return info_L, patch_L

def count_pixel_value(seq, timeL_seq, road_marker, save=False):
    c_R = np.zeros(256)
    c_G = np.zeros(256)
    c_B = np.zeros(256)
    for time in tqdm(timeL_seq):
        if type(road_marker[seq][time]) is not np.ndarray:
            continue
        img = np.array(Image.open(f'./ITRI_dataset/{seq}/dataset/{time}/raw_image.jpg'))
        _, patch = get_bounding_box_region(img, road_marker[seq][time])
        for image_id in range(len(patch)):
            VR, CR = np.unique(patch[image_id][:, :, 0], return_counts=True)
            VG, CG = np.unique(patch[image_id][:, :, 1], return_counts=True)
            VB, CB = np.unique(patch[image_id][:, :, 2], return_counts=True)
            c_R[VR] += CR
            c_G[VG] += CG
            c_B[VB] += CB
    if save:
        plt.figure(figsize=(3*5, 1*5))
        plt.subplot(1, 3, 1)
        plt.plot(c_R)
        plt.title('Count of pixel (R)', fontsize=15)
        plt.subplot(1, 3, 2)
        plt.plot(c_G)
        plt.title('Count of pixel (G)', fontsize=15)
        plt.subplot(1, 3, 3)
        plt.plot(c_B)
        plt.title('Count of pixel (B)', fontsize=15)
        plt.savefig('pixel_distrinution.png')
    else:
        return c_R, c_G, c_B

def dilate_mask(iteration=4, save=True):
    '''
    :param iteration: {int} times of dilate
    :param save: {bool} save result or not
    :return: {dict} return dict of dilated mask if save is False 
    '''
    re = {}
    for file in os.listdir('./ITRI_dataset/camera_info/lucid_cameras_x00'):
        if file[-3:] != 'png':
            continue
        mask = np.array(Image.open(f'./ITRI_dataset/camera_info/lucid_cameras_x00/{file}'))[:, :, :3]
        kernel = np.ones((3,3), np.uint8)
        dilation = cv2.dilate(mask, kernel, iterations = iteration)
        dilation[dilation > 0] = 255
        if save:
            Image.fromarray(dilation).save(file.split('_')[2] + '_dilated_mask.png')
        else:
            re[file.split('_')[2]] = dilation
    if not save:
        return re

def generate_below_mask(camera, save=False):
    '''
    :param camera: {string} camera name (f, fr, fl, b)
    :param save: {bool} save result or not
    :return: {pil image/None} 
    '''
    all_pixel_grid = np.array(np.meshgrid(np.arange(1440), np.arange(928))).reshape(2, -1).T
    all_camera_coor = pixel_to_world(all_pixel_grid, camera)
    f_camera_coor = world_coor_transform(all_camera_coor, camera+'2base', True)

    higher_than_camera_mask = f_camera_coor[:, 1] > 0
    higher_w = all_pixel_grid[higher_than_camera_mask][:, 0]
    higher_h = all_pixel_grid[higher_than_camera_mask][:, 1]

    re_map = np.zeros((928, 1440))
    re_map[higher_h, higher_w] = 255
    if save:
        Image.fromarray(re_map.astype(np.uint8)).save(f'{camera}_below_mask.png')
    else:
        return Image.fromarray(re_map.astype(np.uint8))

def adaptive_mask(img, TYPE, ratio):
    ''''''
    # print('Use adaptive mask')
    if TYPE == 'f':
        car_mask = np.array(Image.open('./mask/f_dilated_mask.png'))
        below_mask = np.array(Image.open('./mask/f_below_mask.png'))
    elif TYPE == 'fl':
        car_mask = np.array(Image.open('./mask/fl_dilated_mask.png'))
        below_mask = np.array(Image.open('./mask/fl_below_mask.png'))
    elif TYPE == 'fr':
        car_mask = np.array(Image.open('./mask/fr_dilated_mask.png'))
        below_mask = np.array(Image.open('./mask/fr_below_mask.png'))
    elif TYPE == 'b':
        car_mask = np.array(Image.open('./mask/b_dilated_mask.png'))
        below_mask = np.array(Image.open('./mask/b_below_mask.png'))
        
    valid_region = np.logical_and(below_mask == 255, car_mask[:, :, 0] == 0).astype(np.uint8)
    valid_coor = np.array(np.where(valid_region == 1)).T
    R, B, G = img[:, :, 0].copy(), img[:, :, 1].copy(), img[:, :, 2].copy()
    Rs = np.sort(R[valid_coor[:, 0], valid_coor[:, 1]]).astype(np.int64)
    Gs = np.sort(G[valid_coor[:, 0], valid_coor[:, 1]]).astype(np.int64)
    Bs = np.sort(B[valid_coor[:, 0], valid_coor[:, 1]]).astype(np.int64)
    length = valid_coor.shape[0]
    # print(ratio)
    intensity_th = (Rs[-1*int(length*ratio)] + Gs[-1*int(length*ratio)] + Bs[-1*int(length*ratio)])/3
    intensity_mask = np.logical_and(np.logical_and(R > intensity_th, G > intensity_th), B > intensity_th)
    
    # white_mask = np.logical_and(np.logical_and(np.abs(R - G) < white_th, 
    #                                            np.abs(R - B) < white_th), 
    #                             np.abs(G - B) < white_th)
    # mask = np.logical_and(intensity_mask, white_mask)
    mask = intensity_mask
    mask = mask.astype(np.uint8)*255
    
    if TYPE == 'f':
        mask[:510, :] = 0
    elif TYPE == 'fl':
        mask[:350, :] = 0
    elif TYPE == 'fr':
        mask[:400, :] = 0
    elif TYPE == 'b':
        mask[:520, :] = 0
    # return mask, intensity_mask, white_mask
    return mask, below_mask, car_mask

def preprocess(img, TYPE, mask_th=128, intensity_ratio=0.1, 
               open_k_size=3, open_iter=1, 
               method='harris', harris_th_ratio=0.01, 
               maxCorners=2000, 
               qualityLevel=0.01, 
               minDistance=10):
    '''Function to preprocess image for better corner detection
    :param img: {numpy.ndarray} image array (IN RGB form)
    :param TYPE: {string} 
    :param mask_th: {int} threshold of filtering three channel
    :param open_k_size: {int} kernel size of opening process
    :param open_iter: {int} doing how much opening
    :return: {tuple} tuple of result of each step (opening, canny, harris, corner, cerner coor)
    '''
    # mask = np.logical_and(np.logical_and(img[:, :, 0] > mask_th, img[:, :, 1] > mask_th), img[:, :, 2] > mask_th)
    # mask = mask.astype(np.uint8)
    # ## rregion above horizon is determined by f>1 b>2 fr>3 fl>296
    # if TYPE == 'f':
    #     mask[:510, :] = 0
    #     car_mask = np.array(Image.open('./mask/f_dilated_mask.png'))
    #     below_mask = np.array(Image.open('./mask/f_below_mask.png'))
    # elif TYPE == 'fl':
    #     # mask[:390, :] = 0
    #     mask[:350, :] = 0
    #     car_mask = np.array(Image.open('./mask/fl_dilated_mask.png'))
    #     below_mask = np.array(Image.open('./mask/fl_below_mask.png'))
    # elif TYPE == 'fr':
    #     mask[:400, :] = 0
    #     car_mask = np.array(Image.open('./mask/fr_dilated_mask.png'))
    #     below_mask = np.array(Image.open('./mask/fr_below_mask.png'))
    # elif TYPE == 'b':
    #     mask[:520, :] = 0
    #     car_mask = np.array(Image.open('./mask/b_dilated_mask.png'))
    #     below_mask = np.array(Image.open('./mask/b_below_mask.png'))

    mask, below_mask, car_mask = adaptive_mask(img, TYPE, ratio=intensity_ratio)
    
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k_size,  open_k_size))
    open_re = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=open_iter)
    canny_re = cv2.Canny(open_re*255, 30, 150)

    if method == 'harris':
        harris_re = cv2.cornerHarris(open_re*255, 2, 3, 0.04)
        harris_re = cv2.dilate(harris_re, None)
        harris_re = (harris_re > harris_th_ratio*harris_re.max())*255
        # harris_re *= np.logical_not(car_mask.astype(np.bool)).astype(np.uint8)[:, :, 0]
        corner_array = np.array(np.where(harris_re == 255)).T
    if method == 'shi_tomasi':
        # gray = cv2.cvtColor(open_re, cv2.COLOR_BGR2GRAY)
        corner_array = cv2.goodFeaturesToTrack(open_re, maxCorners, qualityLevel, minDistance)[:, 0, :].astype(np.int64)[:, ::-1]
    corner_re = img.copy()
    filt = []
    for i in range(corner_array.shape[0]):
        # print(car_mask)
        if car_mask[corner_array[i, 0], corner_array[i, 1], 0] == 255:
            filt.append(False)
            continue
        if below_mask[corner_array[i, 0], corner_array[i, 1]] == 0:
            filt.append(False)
            continue
        filt.append(True)
        cv2.circle(corner_re, (corner_array[i, 1], corner_array[i, 0]), 3, (255, 0, 0), -1)
    corner_array = corner_array[np.array(filt)][:, ::-1]
    # corner_array[:, 1] = img.shape[0] - 1 - corner_array[:, 1]
    return open_re, canny_re, corner_re, corner_array

def plot_pre_process_re(ori, TYPE, mask_th=128, open_k_size=3, open_iter=1, method='harris', harris_th_ratio=0.01, filename=''):
    '''Function to show one image's result of preprocessing
    :param img: {numpy.ndarray} image array (IN RGB form)
    :param TYPE: {string} 
    :param mask_th: {int} threshold of filtering three channel
    :param open_k_size: {int} kernel size of opening process
    :param open_iter: {int} doing how much opening
    :param filename: {string} save filename
    :return: None
    '''
    A, B, C, D = preprocess(ori, TYPE, mask_th, open_k_size, open_iter, method, harris_th_ratio)
    plt.figure(figsize=(2*10, 2*8))
    plt.subplot(2, 2, 1)
    plt.imshow(ori)
    plt.subplot(2, 2, 2)
    plt.imshow(A, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(B, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(C, cmap='gray')
    plt.tight_layout()
    if filename == '':
        plt.show()
    else:
        plt.savefig(filename)

def correct_to_land(world_array, filt=True, method='euclidean', filt_dist=10):
    '''Function to set height (index equal to 1) into -1.63 (theoretical ground)
    :param world_array: {numpy.ndarray} original point cloud
    :param filt: {bool} use distance to filter potential wrong point
    :param dist: {int} 
    '''
    scalar = -1*1.63/world_array[:, 2]
    correct_array = world_array*np.repeat(scalar[:, np.newaxis], 3, axis=1)
    if filt:
        if method == 'manhattan':
            mask = np.logical_and(correct_array[:, 0] < filt_dist, correct_array[:, 1] < filt_dist)
        elif method == 'euclidean':
            mask = (np.sqrt(correct_array[:, 0]**2 + correct_array[:, 1]**2) < filt_dist)
        # elif method == other_method:
        correct_array = correct_array[mask]
    return correct_array

def validate_top_view_of_F(choose, DIR='ITRI_DLC', seq=None, time=None, save=False):
    '''
    TODO
    '''
    test_camera_dict = pickle_load('./camera_dict_test.pkl')
    if seq is None and time is not None:        
        if choose == 'fl':
            img = np.array(Image.open(f'./ITRI_DLC/test1/dataset/1681710758_207223657/raw_image.jpg'))
            camera = test_camera_dict['test1']['1681710758_207223657'].split('_')[-2]
        elif choose == 'f':
            img = np.array(Image.open(f'./ITRI_DLC/test1/dataset/1681710752_188189800/raw_image.jpg'))
            camera = test_camera_dict['test1']['1681710752_188189800'].split('_')[-2]
        elif choose == 'fr':
            img = np.array(Image.open(f'./ITRI_DLC/test1/dataset/1681710759_661666137/raw_image.jpg'))
            camera = test_camera_dict['test1']['1681710759_661666137'].split('_')[-2]
    else:
        img = np.array(Image.open(f'./{DIR}/{seq}/dataset/{time}/raw_image.jpg'))
        camera = choose 
    _, _, _, corner_point = preprocess(img, camera, method='shi_tomasi')
    point_cloud = pixel_to_world(corner_point, camera)
    point_cloud = world_coor_transform(point_cloud, camera+'2base', True)
    point_cloud = correct_to_land(point_cloud, filt_dist=20)
    
    plt.scatter(point_cloud[:, 0], point_cloud[:, 2], s=1)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.scatter(0, 0)
    if save:
        plt.savefig('./val_example.png')
    else:
        plt.show()

def get_nearst_assist_time(time, ori_camera, timestamp_L, camera_L, th):
    '''
    :param time: input time 
    :param ori_camera: {string} input time's camera
    :param timestamp_L: {list} list of time stamp
    :param camera_L: {list} list of camera info of all timestamp
    :param th: {float} threshold of getting assist time stamp
    :return: {list} list of assist time stamp
    '''
    camera_candidate = ['f', 'fl', 'fr', 'b']
    camera_candidate.remove( ori_camera )
    # print(camera_candidate)
    numeric_time = int(time.split('_')[0]) + int(time.split('_')[1])/(10**9)
    assist = []
    
    numeric_timestamp_L = []
    for t in timestamp_L:
        numeric_timestamp_L.append(  int(t.split('_')[0]) + int(t.split('_')[1])/(10**9)  )
    numeric_timestamp_L = np.array(numeric_timestamp_L)
    
    for c in camera_candidate:
        temp_sub_index = np.argmin(np.abs(numeric_timestamp_L[camera_L == c] - numeric_time))
        if abs(numeric_timestamp_L[camera_L == c][temp_sub_index] - numeric_time) > th:
            continue
        temp_index = np.where(numeric_timestamp_L == numeric_timestamp_L[camera_L == c][temp_sub_index])[0][0]
        
        assist.append(  timestamp_L[temp_index]  )
    return assist

###########################################################################################################

def get_localization_timestamp():
    ''''''
    re = {}
    with open('./ITRI_dataset/seq1/localization_timestamp.txt', 'r') as f:
        seq1 = f.readlines()
    with open('./ITRI_dataset/seq2/localization_timestamp.txt', 'r') as f:
        seq2 = f.readlines()
    with open('./ITRI_dataset/seq3/localization_timestamp.txt', 'r') as f:
        seq3 = f.readlines()
    with open('./ITRI_DLC2/seq1/localization_timestamp.txt', 'r') as f:
        seq1_new = f.readlines()
    with open('./ITRI_DLC2/seq2/localization_timestamp.txt', 'r') as f:
        seq2_new = f.readlines()
    with open('./ITRI_DLC2/seq3/localization_timestamp.txt', 'r') as f:
        seq3_new = f.readlines()
    
    with open('./ITRI_DLC/test1/localization_timestamp.txt', 'r') as f:
        test1 = f.readlines()
    with open('./ITRI_DLC/test2/localization_timestamp.txt', 'r') as f:
        test2 = f.readlines()

    with open('./ITRI_DLC2/test1/localization_timestamp.txt', 'r') as f:
        test1_new = f.readlines()
    with open('./ITRI_DLC2/test2/localization_timestamp.txt', 'r') as f:
        test2_new = f.readlines()
        
    re['seq1'] = seq1
    re['seq2'] = seq2
    re['seq3'] = seq3
    
    re['seq1_new'] = seq1_new
    re['seq2_new'] = seq2_new
    re['seq3_new'] = seq3_new
    
    re['test1'] = test1
    re['test2'] = test2
    re['test1_new'] = test1_new
    re['test2_new'] = test2_new
    
    return re

def ICP(source, target, threshold, init_pose, iteration=30):
    # implement iterative closet point and return transformation matrix
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_pose,
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration)
    )
    # print(reg_p2p)
    assert len(reg_p2p.correspondence_set) != 0, 'The size of correspondence_set between your point cloud and sub_map should not be zero.'
    # print(reg_p2p.transformation)
    return reg_p2p.transformation

def csv_reader(filename):
    # read csv file into numpy array
    data = np.loadtxt(filename, delimiter=',')
    return data

def numpy2pcd(arr):
    # turn numpy array into open3d point cloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd

def generate_pred_pose(name, re_dict, sub_map_dict, init_pose_dict, th=10, mode='train'):
    '''
    
    '''
    if mode == 'train':
        re = {}
        for seq in ['seq1', 'seq2', 'seq3']:
            re[seq] = ''
            seq_time = txt_load(f'./ITRI_DLC2/{seq}/localization_timestamp.txt')
            for time in tqdm(seq_time):
                target_pcd = numpy2pcd(sub_map_dict[seq][time])
                source_pcd = numpy2pcd(re_dict[seq][time])
                init_pose = init_pose_dict[seq][time]
                try:
                    transformation = ICP(source_pcd, target_pcd, threshold=th, init_pose=init_pose)
                except:
                    print(time)
                    re[seq] += str(pred_x) + ' ' + str(pred_y) + '\n'
                    continue
                pred_x = transformation[0,3]
                pred_y = transformation[1,3]
                re[seq] += str(pred_x) + ' ' + str(pred_y) + '\n'
        with open(f'./result/{name}/solution/seq1/pred_pose.txt', 'w') as f:
            f.write(re['seq1'])
        with open(f'./result/{name}/solution/seq2/pred_pose.txt', 'w') as f:
            f.write(re['seq2'])
        with open(f'./result/{name}/solution/seq3/pred_pose.txt', 'w') as f:
            f.write(re['seq3'])
    elif mode == 'test':
        re = {}
        for test in ['test1', 'test2']:
            re[test] = ''
            test_time = txt_load(f'./ITRI_DLC2/{test}/localization_timestamp.txt')
            for time in tqdm(test_time):
                target_pcd = numpy2pcd(sub_map_dict[test][time])
                source_pcd = numpy2pcd(re_dict[test][time])
                init_pose = init_pose_dict[test][time]
                try:
                    transformation = ICP(source_pcd, target_pcd, threshold=th, init_pose=init_pose)
                except:
                    print(time)
                    re[test] += str(pred_x) + ' ' + str(pred_y) + '\n'
                    continue
                pred_x = transformation[0,3]
                pred_y = transformation[1,3]
                re[test] += str(pred_x) + ' ' + str(pred_y) + '\n'
        with open(f'./result/{name}/solution/test1/pred_pose.txt', 'w') as f:
            f.write(re['test1'])
        with open(f'./result/{name}/solution/test2/pred_pose.txt', 'w') as f:
            f.write(re['test2'])
    return 'Done'

###########################################################################################################

def harris_fn(img,plot=True):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Turn image to gray
    harris = cv2.cornerHarris(gray,2,3,0.04)  # Applies harris corner detector to gray image
    #harris = cv2.cornerHarris(gray,5,5,0.05) 
    
    # Result is dilated for marking the corners, not important
    harris = cv2.dilate(harris,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    threshold= 0.01*harris.max()

    corner_point = np.argwhere(harris >threshold)#.tolist()
    
    img[harris>threshold]=[0,0,255]
    num_corners = np.sum(harris > threshold)
    
    if plot:
        plt.figure("Harris detector")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
    return corner_point, num_corners

def intrinsic_transform(homogeneous_matrix, homogeneous_coor):
    '''
    :param homogeneous_matrix: {numpy.ndarray} homogeneous_matrix
    :param pixel_coor: {numpy.ndarray} dimension: (N*4) 
    :return: {numpy.ndarray} result camera coordinate vector (x, y, z)
    '''
    # 2D -> 3D
    homogeneous_3d_coor = np.dot(np.linalg.inv(homogeneous_matrix), homogeneous_coor)
    camera_coor = homogeneous_3d_coor[:3] / homogeneous_3d_coor[3]
    return camera_coor
    
def pinhole_model(dst_coor, c_type=''):
    '''
    :param dst_coor: {numpy.ndarray} pixel coordinate vector (x_p, y_p)
    :param c_type: {string} camera type ('b' or 'f' or 'fl', or 'fr')
    :return: {numpy.ndarray} result coordinate vector (x, y, z)
    '''
    global D_B, D_F, D_FR, D_FL, K_B, K_F, K_FR, K_FL, P_B, P_F, P_FR, P_FL
    # x_p, y_p = list(dst_coor[:, 0]), list(dst_coor[:, 1])
    # homogeneous_coor = np.squeeze(np.dstack((x_p, y_p, np.ones(len(x_p)), np.ones(len(x_p)))), axis=0)
    # ST: (y, x)
    # Harris: (x, y)
    x_p, y_p = dst_coor[:, 1], dst_coor[:, 0]
    homogeneous_coor = np.vstack((x_p, y_p, np.ones(len(x_p)), np.ones(len(x_p))))
    if c_type == 'b':
        homogeneous_matrix = np.vstack((P_B, [0, 0, 0, 1]))
        e_type = 'b2base'
    elif c_type == 'f':
        homogeneous_matrix = np.vstack((P_F, [0, 0, 0, 1]))
        e_type = 'f2base'
    elif c_type == 'fl':
        homogeneous_matrix = np.vstack((P_FL, [0, 0, 0, 1]))
        e_type = 'fl2base'
    elif c_type == 'fr':
        homogeneous_matrix = np.vstack((P_FR, [0, 0, 0, 1]))
        e_type = 'fr2base'
    camera_coor = intrinsic_transform(homogeneous_matrix, homogeneous_coor).T
    world_coor = [transform_1coor(c_c, TYPE=e_type) for c_c in camera_coor]
    return world_coor
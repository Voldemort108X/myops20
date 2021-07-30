import os
import argparse
import nibabel as nib
import distutils
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat,savemat
from helper import *

def func_createImageList(path_data):
    path_train_img = os.path.join(path_data,'train25')
    path_train_lab = os.path.join(path_data,'train25_myops_gd')
    lab_names = os.listdir(path_train_lab)
    
    img_C0_list = list(map(lambda lab_name: np.array(nib.load(os.path.join(path_train_img,lab_name.replace('gd','C0'))).get_data()), lab_names))
    img_DE_list = list(map(lambda lab_name: np.array(nib.load(os.path.join(path_train_img,lab_name.replace('gd','DE'))).get_data()), lab_names))
    img_T2_list = list(map(lambda lab_name: np.array(nib.load(os.path.join(path_train_img,lab_name.replace('gd','T2'))).get_data()), lab_names))
    lab_list = list(map(lambda lab_name: np.array(nib.load(os.path.join(path_train_lab,lab_name)).get_data()), lab_names))
    
    return img_C0_list, img_DE_list, img_T2_list, lab_list
    
def func_applyRandomWarping(input_list, transmap_list):
    nitr = len(transmap_list)
    assert nitr > 1
    augmented_input_list = []
    for i in range(nitr):
        augmented_input_list_curritr = list(map(lambda image, transmap: func_applyElasticTransformParallel(image, transmap), input_list, transmap_list[i]))
        augmented_input_list.append(augmented_input_list_curritr)
    return [input_list] + augmented_input_list

def func_generateRandomWarpingTransmap(input_list, kernal_size, kernal_range, window_size, nitr):
    assert nitr > 1
    numCase = int(len(input_list))
    transmap_list = []
    for i in range(nitr-1):
        transmap_list_curritr = []
        for j in range(numCase):
            transmap = func_generateTransmap(input_list[j], kernal_size, kernal_range)
            weightMatrix = func_generateWeightMatrix(input_list[j], window_size)
            transmap = func_applyWeightMap(transmap, weightMatrix)
            transmap_list_curritr.append(transmap)
        transmap_list.append(transmap_list_curritr)
    return transmap_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default=os.path.join(os.getcwd(),'Data','Original_data'), help='The path to raw data (should contain subfolders: train25, train25_myops_gd, test20)')
    parser.add_argument('--destpath', default=os.path.join(os.getcwd(),'Data','Augmented_data','train_25'), help='Destination path to save augmented input and label list')
    parser.add_argument('--ks', default=8, type=int, help='kernel size for upsampling matrix')
    parser.add_argument('--kr', default=5, type=int, help='kernal range for the element in kernel with [-kr,kr]')
    parser.add_argument('--ws', default=40, type=int, help='window size for the gradually decay in boundary to reduce stretching artifact')
    parser.add_argument('--nitr',default=5 , type=int, help='number of random warping augmentation')
    args = parser.parse_args()

    img_C0_list, img_DE_list, img_T2_list, lab_list = func_createImageList(args.datapath)
    transmap_list = func_generateRandomWarpingTransmap(img_C0_list, args.ks, args.kr, args.ws, args.nitr)
    augmented_img_C0_list = func_applyRandomWarping(img_C0_list, transmap_list)
    augmented_img_DE_list = func_applyRandomWarping(img_DE_list, transmap_list)
    augmented_img_T2_list = func_applyRandomWarping(img_T2_list, transmap_list)
    augmented_lab_list = func_applyRandomWarping(lab_list, transmap_list)

    if not os.path.exists(args.destpath):
        os.makedirs(args.destpath)

    np.save(os.path.join(args.destpath,'augmented_img_C0_list.npy'),np.array(augmented_img_C0_list, dtype=object))
    np.save(os.path.join(args.destpath,'augmented_img_DE_list.npy'),np.array(augmented_img_DE_list, dtype=object))
    np.save(os.path.join(args.destpath,'augmented_img_T2_list.npy'),np.array(augmented_img_T2_list, dtype=object))
    np.save(os.path.join(args.destpath,'augmented_lab_list.npy'),np.array(augmented_lab_list, dtype=object))

    # jupyternotebook debug
    # img_C0_list, img_DE_list, img_T2_list, lab_list = func_createImageList(os.path.join(os.getcwd(),'Data','Debug_data'))
    # transmap_list = func_generateRandomWarpingTransmap(img_C0_list, 8, 5, 40, 3)




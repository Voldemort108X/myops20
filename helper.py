import os
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import random
from config import *

def func_findParentPath(org_path,num_itr=1):
    parentPath = org_path
    for i in range(num_itr):
        parentPath = os.path.abspath(os.path.join(parentPath,os.pardir))
    return parentPath

def func_generateTransmap(input_image,kernel_size,range):
    """
    # Generate transmap using upsampling from a random uniformly distributed matrix
    :param input_image: input image
    :param kernel_size: kernel size for upsampling matrix
    :param range: range for the element in kernel with [-range,range]
    :return: transmap for the input image with size [img_size,img_size,2]
    """
    [img_row, img_col] = [input_image.shape[0], input_image.shape[1]]

    transmap_ni = np.zeros((img_row,img_col,2))
    kernel = -range + 2*range*np.random.rand(kernel_size,kernel_size,2)

    factor_row = img_row/kernel_size
    factor_col = img_col/kernel_size

    # bilinear interpolation for image upsampling
    transmap_ni[:, :, 0] = zoom(kernel[:, :, 0], [factor_row, factor_col], order=1)
    transmap_ni[:, :, 1] = zoom(kernel[:, :, 1], [factor_row, factor_col], order=1)

    return transmap_ni

def func_generateWeightMatrix(input_image,window_size):
    """
    # Generate weight matrix to eliminate boundary stretch artifact
    :param input_image: input image
    :param window_size: window size for the gradually decay in boundary to reduce stretching artifact
    :return: the weight matrix for transmap with size (img_size,img_size)
    """
    # initialize weight matrix with "0"
    [img_row, img_col] = [input_image.shape[0], input_image.shape[1]]
    weight_matrix = np.zeros((img_row,img_col))

    # fill the weight matrix with "1" in the center bounded by window_size
    for i in np.arange(window_size,img_row-window_size):
        for j in np.arange(window_size,img_col-window_size):
            weight_matrix[i,j] = 1

    # gradually descend the boundary from "1" to "0"
    coef = 1/window_size
    for step in range(1,window_size+1):
        # strip 1: horizontal top
        weight_matrix[np.arange(window_size - step, img_row - window_size + step - 1), window_size - step] = 1 - step * coef
        # strip 2: horizontal bottom
        weight_matrix[np.arange(window_size - step, img_row - window_size + step - 1), img_col - window_size + step - 1] = 1 - step * coef
        # strip 3: vertical left
        weight_matrix[window_size - step, np.arange(window_size - step, img_col - window_size + step - 1)] = 1 - step * coef
        # strip 4: verticle right
        weight_matrix[img_row - window_size + step - 1, np.arange(window_size - step, img_col - window_size + step - 1)] = 1 - step * coef

    return weight_matrix

def func_applyElasticTransform(img,transmap):
    """
    # apply elastric tranformation
    :param img: input image with size [img_size,img_size,num_channels]
    :param transmap: transmap with size [img_size,img_size,2]
    :return: deformed image with size [img_size,img_size,num_channels] with the same deformation for all channels
    """
    num_channels = img.shape[2]
    img_final = np.zeros(img.shape, dtype=np.int16)

    fTotalTranslationX = transmap[:, :, 0]
    fTotalTranslationY = transmap[:, :, 1]
    [h, w] = fTotalTranslationX.shape
    x = np.arange(w)
    y = np.arange(h)
    [x2, y2] = np.meshgrid(x, y)
    xp = x2 + fTotalTranslationX
    yp = y2 + fTotalTranslationY

    for i in range(num_channels):
        # img[:, :, i] = (img[:,:,i]-np.min(img[:,:,i]))/(np.max(img[:, :, i])-np.min(img[:,:,i]))
        # print(np.array(x2.flatten()).shape)
        # print(np.array((x2.flatten(),x2.flatten())).shape)
        temp = griddata(np.array((x2.flatten(),y2.flatten())).T,img[:,:,i].flatten(),np.array((xp.flatten(), yp.flatten())).T,method='linear')
        img_final[:, :, i] = np.reshape(temp,(h,w))

    return img_final

def func_applyElasticTransformParallel(img,transmap):
    """
    # apply elastric tranformation with parallel computing
    :param img: input image with size [img_size,img_size,num_channels]
    :param transmap: transmap with size [img_size,img_size,2]
    :return: deformed image with size [img_size,img_size,num_channels] with the same deformation for all channels
    """
    num_channels = img.shape[2]
    print('debug: num_channels', num_channels)
    img_final = np.zeros(img.shape, dtype=np.int16)

    # plt.imshow(img[:,:,0])
    # plt.show()

    fTotalTranslationX = transmap[:, :, 0]
    fTotalTranslationY = transmap[:, :, 1]
    [h, w] = fTotalTranslationX.shape
    x = np.arange(w)
    y = np.arange(h)
    [x2, y2] = np.meshgrid(x, y)
    xp = x2 + fTotalTranslationX
    yp = y2 + fTotalTranslationY

    np.savez('temp',translation_data=[x2,y2,xp,yp])
    data = np.load('temp.npz')['translation_data']

    img_list = []
    for i in range(num_channels):
        #img[:,:,i] = (img[:,:,i] - np.min(img[:,:,i])) / (np.max(img[:,:,i]) - np.min(img[:,:,i]))
        img_list.append(img[:,:,i].flatten())

    num_cores = multiprocessing.cpu_count()
    processed_img = Parallel(n_jobs=num_cores)(delayed(applyTransform)(img,data) for img in img_list)

    cnt = 0
    for element in processed_img:
        test_1 = np.array(element)
        test_2 = test_1[0]
        img_final[:,:,cnt] = test_2
        cnt = cnt + 1

    # print('debug: img_final shape',img_final.shape)
    # plt.imshow(img_final[:,:,0])
    # plt.show()

    return img_final

def applyTransform(img_list,data):
    """
    :param img_list: flattened image list with shape img_size*img_size
    :param data: translation data zip file with shape [4,img_size,img_size]
    :return: deformed image list with length num_channels, each element has a shape of [1,img_size,img_size]
    """
    img_final_list = []

    x2 = data[0]
    y2 = data[1]
    xp = data[2]
    yp = data[3]

    [h, w] = x2.shape
    temp = griddata(np.array((x2.flatten(), y2.flatten())).T, img_list,
                        np.array((xp.flatten(), yp.flatten())).T, method='linear')
    img_final = np.reshape(temp, (h, w))
    img_final_list.append(img_final)

    return img_final_list

def func_applyWeightMap(transmap, weightMatrix):
    """
    :param transmap: translation map to deform the image
    :param weightMatrix: weight matrix to reduce the boundary stretching artifact
    :return: improved translation map
    """
    transmap[:,:,0] = np.multiply(transmap[:,:,0], weightMatrix)
    transmap[:,:,1] = np.multiply(transmap[:,:,1], weightMatrix)
    
    return transmap

def func_imgCrop(image,crop_size):
    # input image is in grayscale
    size_x,size_y = image.shape
    return image[int(int(size_x/2)-crop_size/2):int(int(size_x/2)+crop_size/2),int(int(size_y/2)-crop_size/2):int(int(size_y/2)+crop_size/2)]

def func_imgNormalize(image):
    im_min, im_max = np.percentile(image,[5,95])
    return np.clip(np.array((image-im_min)/(im_max-im_min), dtype=np.float32), 0.0, 1.0)

def func_linearInputEncoder(img_C0_list, img_DE_list, img_T2_list, moduleIndex):
    if moduleIndex == class_values.index(LV_BP):
        # LV_BP block uses bSSFP as input
        return img_C0_list
    if moduleIndex == class_values.index(RV_BP):
        # RV_BP block uses bSSFP as input
        return img_C0_list
    if moduleIndex == class_values.index(LV_NM):
        # LV_Epicardium block uses bSSFP as input
        return img_C0_list
    if moduleIndex == class_values.index(LV_ME):
        # LV_MEMS block uses LGE as input
        return img_DE_list
    if moduleIndex == class_values.index(LV_MS):
        # LV_MS block uses T2 as input
        return img_T2_list

def func_linearLabelEncoder(patch_size, moduleIndex, mask_):
    mask_o = np.zeros((patch_size,patch_size,1), dtype=np.float32)
    if moduleIndex == class_values.index(LV_BP):
        # LV_BP block uses LV_BP as target
        mask_o = mask_[:,:,moduleIndex]
    if moduleIndex == class_values.index(RV_BP):
        # RV_BP block uses RV_BP as target
        mask_o = mask_[:,:,moduleIndex]
    if moduleIndex == class_values.index(LV_NM):
        # LV_Epicardium block uses LV_BP+LV_NM+LV_ME+LV_MS as target
        print('adding mask LV_ME, LV_MS, LV_BP to LV_NM')
        mask_o = mask_[:,:,moduleIndex] + mask_[:,:,class_values.index(LV_BP)] + mask_[:,:,class_values.index(LV_ME)] + mask_[:,:,class_values.index(LV_MS)]
    if moduleIndex == class_values.index(LV_ME):
        # LV_MEMS block uses LV_ME+LV_MS as target
        mask_o = mask_[:,:,moduleIndex] + mask_[:,:,class_values.index(LV_MS)]
    if moduleIndex == class_values.index(LV_MS):
        # LV_MS block uses LV_MS as target
        print('adding nothing to LV_MS')
        mask_o = mask_[:,:,moduleIndex]

    return mask_o

def func_saveTrainingDataIntoSlices(destPath, train_inputs, train_labels, mode):
    numOfData = len(train_inputs)
    destPath_input = os.path.join(destPath, mode, 'input')
    destPath_label = os.path.join(destPath, mode, 'label')
    if not os.path.exists(destPath_input):
        os.makedirs(destPath_input)
    if not os.path.exists(destPath_label):
        os.makedirs(destPath_label)

    for i in range(numOfData):
        np.save(os.path.join(destPath_input, str(i+1)+'.npy'), train_inputs[i])
        np.save(os.path.join(destPath_label, str(i+1)+'.npy'), train_labels[i])

def func_createLabelToMask(label,myo_type):
    size_x,size_y = label.shape
    mask = np.zeros(shape=(size_x,size_y))

    if myo_type == 'LV_MS' or myo_type == 'BG':
        indices = np.where(label==class_values_mrcnn_dict[myo_type])
        x = indices[0]
        y = indices[1]
        for j in range(indices[0].shape[0]):
            mask[(x[j],y[j])] = 1
        
        return mask

    if myo_type == 'LV_ME':
        indices = np.where(label==class_values_mrcnn_dict[myo_type])
        indices_LVMS = np.where(label==LV_MS)
        x = indices[0]
        y = indices[1]
        for j in range(indices[0].shape[0]):
                #print(x[j],y[j])
            mask[(x[j],y[j])] = 1
            
        x_ = indices_LVMS[0]
        y_ = indices_LVMS[1]
        for z in range(indices_LVMS[0].shape[0]):
                #print(x[j],y[j])
            mask[(x_[z],y_[z])] = 1

        return mask
# %%
# %%

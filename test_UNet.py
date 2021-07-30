# %%
from matplotlib.pyplot import axis
from helper import func_linearInputEncoder, func_imgCrop, func_imgNormalize
from config import *
from network.UNet import UNet_model
import os
import numpy as np
import tensorflow as tf
import nibabel as nib
import argparse

def func_createTestImageList(path_data):
    path_train_img = os.path.join(path_data,'test20') 
    train_names = os.listdir(path_train_img)
    pseudo_lab_names = list(map(lambda C0_img_name: C0_img_name.replace('C0','gd'), list(filter(lambda img_name: 'C0' in img_name,  train_names))))
    
    img_C0_list = list(map(lambda lab_name: np.array(nib.load(os.path.join(path_train_img,lab_name.replace('gd','C0'))).get_data()), pseudo_lab_names))
    img_DE_list = list(map(lambda lab_name: np.array(nib.load(os.path.join(path_train_img,lab_name.replace('gd','DE'))).get_data()), pseudo_lab_names))
    img_T2_list = list(map(lambda lab_name: np.array(nib.load(os.path.join(path_train_img,lab_name.replace('gd','T2'))).get_data()), pseudo_lab_names))
    
    return img_C0_list, img_DE_list, img_T2_list, pseudo_lab_names

def func_prepareTestingData(patch_size, inputCase, test_inputs, inputCaseName, test_names):
    numOfLayer = inputCase.shape[2]
    for layerIndex in range(numOfLayer):
        img_slice = func_imgCrop(inputCase[:,:,layerIndex], patch_size)
        img_slice = np.expand_dims(img_slice, axis=2)
        img_slice = func_imgNormalize(img_slice)
        test_inputs.append(img_slice)
        test_names.append(inputCaseName.split('.')[0]+'_layer_'+str(layerIndex+1))

    return test_inputs, test_names

def func_loadTestingData(datapath, patch_size, moduleIndex):
    img_C0_list, img_DE_list, img_T2_list, pseudo_lab_names = func_createTestImageList(datapath)

    test_inputs, test_names = [], []
    numOfCases = len(img_C0_list)

    for caseIndex in range(numOfCases):
        input_singlechannel_list = func_linearInputEncoder(img_C0_list, img_DE_list, img_T2_list, moduleIndex)
        test_inputs, test_names = func_prepareTestingData(patch_size, inputCase=input_singlechannel_list[caseIndex], test_inputs=test_inputs, inputCaseName=pseudo_lab_names[caseIndex], test_names=test_names)
    
    test_inputs = np.stack(test_inputs, axis=0)

    print('testing input shape', test_inputs.shape)

    return test_inputs, test_names

def func_saveTestPrediction(predis_path, test_predis, test_names):
    numOfTestSlices = len(test_names)
    for sliceIndex in range(numOfTestSlices):
        np.save(os.path.join(predis_path, test_names[sliceIndex].replace('gd','predi')+'.npy'), test_predis[sliceIndex,:,:,:])

def test(dataPath, destPath, modelPath, moduleIndexList, batch_size, patch_size, num_epoch, validation_split, learning_rate):
    #debug_test_inputs, debug_test_names = [], []
    for moduleIndex in moduleIndexList:
        test_inputs, test_names = func_loadTestingData(dataPath, patch_size, moduleIndex)
        
        # debug_test_inputs.append(test_inputs), debug_test_names.append(test_names)
        model = UNet_model(patch_size, patch_size, learning_rate)
        model_name = f"model_UNet_{class_names[moduleIndex]}_bs_{batch_size}_ps_{patch_size}_epoch_{num_epoch}_valid_split_{validation_split}_lr_{learning_rate}"
        weights_path = os.path.join(modelPath, model_name, '{:03d}.h5'.format(num_epoch))
        model.load_weights(weights_path)

        test_predis = model.predict(test_inputs)
        predis_path = os.path.join(destPath, model_name)
        if not os.path.exists(predis_path):
            os.makedirs(predis_path)
        func_saveTestPrediction(predis_path, test_predis, test_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default=os.path.join(os.getcwd(),'Data','Original_data'), help='Test data path for UNet')
    parser.add_argument('--datadest', default=os.path.join(os.getcwd(),'Prediction', 'UNet'), help='Prediction path for UNet')
    parser.add_argument('--modelpath', default=os.path.join(os.getcwd(), 'Models', 'Model_UNet', help='Model path for UNet'))
    parser.add_argument('--moduleIndexList', default=[0, 1, 2, 3, 4], help='model list for testing, 0: LV_BP, 1:RV_BP, 2:LV_NM, 3:LV_ME, 4:LV_MS')
    parser.add_argument('--bs', default=8, type=int, help='batch size for UNet testing model')
    parser.add_argument('--ps', default=256, type=int, help='patch size for UNet testing model')
    parser.add_argument('--epoch', default=500, type=int, help='epoch for UNet testing model')
    parser.add_argument('--vs', default=0.8, help='validation split for UNet testing model')
    parser.add_argument('--lr', default=1e-5, help='learning rate for UNet testing model')
    args = parser.parse_args()

    # jupyternotebook debug
    # test(os.path.join(os.getcwd(),'Data','Original_data'), os.path.join(os.getcwd(),'Prediction', 'UNet'), os.path.join(os.getcwd(),'Models','Model_UNet'), [0,1,2,3,4], 8, 256, 5, 0.8, 1e-5)

#%%


from skimage.measure import label, regionprops
from skimage import exposure
from scipy.ndimage.morphology import binary_fill_holes
import nibabel as nib
import os
import matplotlib.pyplot as plt
import numpy as np
from config import *
import argparse

def extract_predi(predi):
    # input predi with shape HxWx5 and return with a shape HxW
    row,col,n_class = predi.shape
    predi_final = np.zeros((row,col))
    for i in range(n_class):
        indices = np.where(predi[:,:,i]==1)
        x_ = indices[0]
        y_ = indices[1]
        for j in range(indices[0].shape[0]):
            predi_final[(x_[j],y_[j])] = class_values[i]

    return predi_final

def func_prediThreshold(mask_predi,th):
    mask_predi = np.where(mask_predi<=th,0,1)
    return mask_predi

def func_getLargestComponent(bw):
    l = label(np.squeeze(bw))
    #l = label(bw[:,:,0])
    print(np.max(l))
    r = regionprops(l, cache=False) # l is from previous approach
    return (l==(1+np.argmax([i.area for i in r]))).astype(int)

def func_fillHoles(lbl):
    #print(lbl.shape)
    lbl = binary_fill_holes(lbl)

    #for i in range(lbl.shape[2]):
    #    lbl[:,:,i] = binary_fill_holes(lbl[:,:,i])
    return lbl.astype(int)


def func_getNiftyTestInputNames(dataPath):
    nifty_names_input = os.listdir(dataPath)
    nifty_names_predi = []
    for name in nifty_names_input:
        if 'C0' in name:
            nifty_names_predi.append(name.replace('C0','seg'))

    return nifty_names_predi

def func_postProcessSingleSlice(predi_slice, threshold):
    predi_slice = func_prediThreshold(predi_slice, threshold)
    predi_slice = func_getLargestComponent(predi_slice)
    predi_slice = func_fillHoles(predi_slice)
    predi_slice = np.expand_dims(predi_slice, axis=2)

    return predi_slice

def func_maskDecoder(mask_, patch_size):
    # the mask_ should be a shape of [256,256,5] after concatenation with separate U-net predictions
    
    mask_o = np.zeros((patch_size,patch_size,num_classes))
    
    mask_o[:,:,class_values.index(LV_BP)] = mask_[:,:,class_values.index(LV_BP)]
    mask_o[:,:,class_values.index(RV_BP)] = mask_[:,:,class_values.index(RV_BP)]
    mask_o[:,:,class_values.index(LV_NM)] = mask_[:,:,class_values.index(LV_NM)] - mask_[:,:,class_values.index(LV_ME)] - mask_[:,:,class_values.index(LV_BP)] #- mask_[:,:,class_values.index(LV_MS)]
    mask_o[:,:,class_values.index(LV_ME)] = mask_[:,:,class_values.index(LV_ME)] - mask_[:,:,class_values.index(LV_MS)]
    mask_o[:,:,class_values.index(LV_MS)] = mask_[:,:,class_values.index(LV_MS)]

    return mask_o

def func_linearPrediDecoder(predi_slice, threshold, patch_size):
    LV_BP_mask = func_prediThreshold(predi_slice[:,:,0], threshold)
    LV_NM_mask = func_prediThreshold(predi_slice[:,:,2], threshold)
    LV_Myo_mask = func_prediThreshold(LV_NM_mask - LV_BP_mask, threshold)

    predi_slice = func_maskDecoder(predi_slice, patch_size)

    predi_slice[:,:,class_values.index(LV_ME)] = np.multiply(predi_slice[:,:,class_values.index(LV_ME)],LV_Myo_mask)
    predi_slice[:,:,class_values.index(LV_MS)] = np.multiply(predi_slice[:,:,class_values.index(LV_MS)],LV_Myo_mask)

    for layer_index in range(num_classes):
        predi_slice[:,:,layer_index] = func_prediThreshold(predi_slice[:,:,layer_index], threshold)
    
    return predi_slice

def func_createSegFileFromBinaryMask(predi):
    # input predi with shape HxWx5 and return with a shape HxW
    row, col, n_class = predi.shape
    predi_final = np.zeros((row,col))
    for i in range(n_class):
        indices = np.where(predi[:,:,i]==1)
        x_ = indices[0]
        y_ = indices[1]
        for j in range(indices[0].shape[0]):
            predi_final[(x_[j],y_[j])] = class_values[i]

    return predi_final

def func_postProcessSingleCase(dataPath, prediRoot, predi_name, patch_size, epoch_UNet, epoch_MaskRCNN, epoch_UNetPP, threshold=0.5):
    input_name = predi_name.replace('seg', 'C0')
    input_sample = nib.load(os.path.join(dataPath,input_name)).get_data()
    input_header = nib.load(os.path.join(dataPath,input_name)).header

    case_predi = np.zeros(input_sample.shape)
    case_name = predi_name.split('.')[0]

    modelname_UNet_LV_BP = f"model_UNet_LV_BP_bs_8_ps_{patch_size}_epoch_{epoch_UNet}_valid_split_0.8_lr_1e-05"
    modelname_UNet_RV_BP = f"model_UNet_RV_BP_bs_8_ps_{patch_size}_epoch_{epoch_UNet}_valid_split_0.8_lr_1e-05"
    modelname_UNet_LV_NM = f"model_UNet_LV_NM_bs_8_ps_{patch_size}_epoch_{epoch_UNet}_valid_split_0.8_lr_1e-05"
    modelname_UNet_LV_ME = f"model_UNet_LV_ME_bs_8_ps_{patch_size}_epoch_{epoch_UNet}_valid_split_0.8_lr_1e-05"
    modelname_UNet_LV_MS = f"model_UNet_LV_MS_bs_8_ps_{patch_size}_epoch_{epoch_UNet}_valid_split_0.8_lr_1e-05"
    
    modelname_MRCNN_LV_ME = f"model_MaskRCNN_LV_ME_ps_{patch_size}_epoch_{epoch_MaskRCNN}_vs_0.8"
    modelname_MRCNN_LV_MS = f"model_MaskRCNN_LV_MS_ps_{patch_size}_epoch_{epoch_MaskRCNN}_vs_0.8"
    
    modelname_UNetPP_LV_ME = f"model_UNetPP_LV_ME_bs_8_ps_{patch_size}_epoch_{epoch_UNetPP}_valid_split_0.8_lr_1e-05"
    modelname_UNetPP_LV_MS = f"model_UNetPP_LV_MS_bs_8_ps_{patch_size}_epoch_{epoch_UNetPP}_valid_split_0.8_lr_1e-05"

    predi_path_UNet_LV_BP = os.path.join(prediRoot, 'UNet', modelname_UNet_LV_BP)
    predi_path_UNet_RV_BP = os.path.join(prediRoot, 'UNet', modelname_UNet_RV_BP)
    predi_path_UNet_LV_NM = os.path.join(prediRoot, 'UNet', modelname_UNet_LV_NM)
    predi_path_UNet_LV_ME = os.path.join(prediRoot, 'UNet', modelname_UNet_LV_ME)
    predi_path_UNet_LV_MS = os.path.join(prediRoot, 'UNet', modelname_UNet_LV_MS)

    predi_path_MRCNN_LV_ME = os.path.join(prediRoot, 'MaskRCNN', modelname_MRCNN_LV_ME)
    predi_path_MRCNN_LV_MS = os.path.join(prediRoot, 'MaskRCNN', modelname_MRCNN_LV_MS)

    predi_path_UNetPP_LV_ME = os.path.join(prediRoot, 'UNetPP', modelname_UNetPP_LV_ME)
    predi_path_UNetPP_LV_MS = os.path.join(prediRoot, 'UNetPP', modelname_UNetPP_LV_MS)

    test_names = os.listdir(predi_path_UNet_LV_BP)
    cnt_lyr = 0
    for test_slice in test_names:
        if case_name[:-3] in test_names:
            predi_UNet_LV_BP = np.load(os.path.join(predi_path_UNet_LV_BP, test_slice))
            predi_UNet_RV_BP = np.load(os.path.join(predi_path_UNet_RV_BP, test_slice))
            predi_UNet_LV_NM = np.load(os.path.join(predi_path_UNet_LV_NM, test_slice))
            predi_UNet_LV_ME = np.load(os.path.join(predi_path_UNet_LV_ME, test_slice))
            predi_UNet_LV_MS = np.load(os.path.join(predi_path_UNet_LV_MS, test_slice))

            predi_MRCNN_LV_ME = np.load(os.path.join(predi_path_MRCNN_LV_ME, test_slice))
            predi_MRCNN_LV_MS = np.load(os.path.join(predi_path_MRCNN_LV_MS, test_slice))

            predi_UNetPP_LV_ME = np.load(os.path.join(predi_path_UNetPP_LV_ME, test_slice))
            predi_UNetPP_LV_MS = np.load(os.path.join(predi_path_UNetPP_LV_MS, test_slice))

            predi_LV_BP = predi_UNet_LV_BP
            predi_RV_BP = predi_UNet_RV_BP
            predi_LV_NM = predi_UNet_LV_NM
            predi_LV_ME = (predi_UNet_LV_ME + predi_MRCNN_LV_ME + predi_UNetPP_LV_ME)/3
            predi_LV_MS = (predi_UNet_LV_MS + predi_MRCNN_LV_MS + predi_UNetPP_LV_MS)/3

            assert np.max(predi_LV_BP) > threshold
            assert np.max(predi_RV_BP) > threshold
            assert np.max(predi_LV_NM) > threshold
            assert np.max(predi_LV_ME) > threshold
            assert np.max(predi_LV_MS) > threshold

            predi_LV_BP = func_postProcessSingleSlice(predi_LV_BP)
            predi_RV_BP = func_postProcessSingleSlice(predi_RV_BP)
            predi_LV_NM = func_postProcessSingleSlice(predi_LV_NM)
            predi_LV_ME = func_prediThreshold(predi_LV_ME)
            predi_LV_MS = func_prediThreshold(predi_LV_MS)

            predi_slice = np.concatenate((predi_LV_BP, predi_RV_BP, predi_LV_NM, predi_LV_ME, predi_LV_MS), axis=2)
            predi_slice = func_linearPrediDecoder(predi_slice, threshold, patch_size)
            predi_slice = func_createSegFileFromBinaryMask(predi_slice)
            
            size_x, size_y, _ = case_predi.shape
            case_predi[int(int(size_x/2)-patch_size/2):int(int(size_x/2)+patch_size/2),int(int(size_y/2)-patch_size/2):int(int(size_y/2)+patch_size/2),cnt_lyr] = predi_slice
            cnt_lyr += 1
    
    return case_predi, input_header

def func_postprocessing(dataPath, prediRoot, destPath, patch_size, epoch_UNet, epoch_MaskRCNN, epoch_UNetPP):
    nifty_names_predi = func_getNiftyTestInputNames(dataPath)

    if not os.path.exists(destPath):
        os.makedirs(destPath)

    for predi_name in nifty_names_predi:
        case_predi, input_header = func_postProcessSingleCase(dataPath, prediRoot, predi_name, patch_size, epoch_UNet, epoch_MaskRCNN, epoch_UNetPP)
        nfED = nib.nifti1.Nifti1Image(case_predi.astype(np.int16), None, header=input_header.copy())
        print('finished:', predi_name)
        nfED.to_filename(os.path.join(destPath, predi_name))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default=os.path.join(os.getcwd(), 'Data', 'Original_data', 'test20'), help='datapath of original test data')
    parser.add_argument('--prediroot', default=os.path.join(os.getcwd(), 'Prediction'), help='prediction root path for prediction files')
    parser.add_argument('--destpath', default=os.path.join(os.getcwd(),'Prediction','Final_prediction'), help='final prediction files in .nii.gz after post-processing')
    parser.add_argument('--ps', default=256, type=int, help='patch size used in testing')
    parser.add_argument('--epoch_UNet', default=500, type=int, help='number of epoch of UNet model used in testing')
    parser.add_argument('--epoch_MaskRCNN', default=500, type=int, help='number of epoch of UNet model used in testing')
    parser.add_argument('--epoch_UNetPP', default=500, type=int, help='number of epoch of UNet model used in testing')
    args = parser.parse_args()

    func_postprocessing(args.datapath, args.prediroot, args.destpath, args.ps, args.epoch_UNet, args.epoch_MaskRCNN, args.epoch_UNetPP)

    # jupyternotebook debug
    # func_postprocessing(os.path.join(os.getcwd(), 'Data', 'Original_data', 'test20'), os.path.join(os.getcwd(), 'Prediction'), os.path.join(os.getcwd(),'Prediction','Final_prediction'), 256, 5, 5, 5)


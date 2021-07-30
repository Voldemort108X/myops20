# %%
from helper import func_imgNormalize, func_imgCrop, func_imgNormalize, func_linearInputEncoder, func_linearLabelEncoder
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import *
from network.UNet import UNet_model
from tensorflow.keras.callbacks import ModelCheckpoint
import random
import argparse
from scipy.ndimage import rotate

def func_imgPairRandomRotation(img1, img2):
    rand_choice = random.randint(0, 2)
    img1_aug = np.rot90(img1, k=rand_choice+1)
    img2_aug = np.rot90(img2, k=rand_choice+1)
    # vis_aug(img_input,img_input_aug,img_label,img_label_aug)
    return img1_aug, img2_aug

def func_labelToMask(label):
    size_x,size_y = label.shape
    mask = np.zeros(shape=(size_x,size_y,num_classes), dtype=np.float32)
    for i in range(num_classes):
        indices = np.where(label==class_values[i])
        x = indices[0]
        y = indices[1]
        for j in range(indices[0].shape[0]):
            mask[(x[j],y[j])+(i,)] = 1

    return mask

def func_prepareTrainingData(patch_size, moduleIndex, inputCase, train_inputs, labelCase, train_labels):
    numofLayer = inputCase.shape[2]
    for layerIndex in range(numofLayer):
        img_slice = func_imgCrop(inputCase[:,:,layerIndex], patch_size)
        img_slice = np.expand_dims(img_slice, axis=2)
        img_slice = func_imgNormalize(img_slice)
        train_inputs.append(img_slice)

        lab_slice = func_imgCrop(labelCase[:,:,layerIndex], patch_size)
        lab_slice = func_labelToMask(lab_slice)
        lab_slice = func_linearLabelEncoder(patch_size, moduleIndex, lab_slice)
        train_labels.append(lab_slice)

        img_slice_aug, lab_slice_aug = func_imgPairRandomRotation(img_slice, lab_slice)
        train_inputs.append(img_slice_aug)
        train_labels.append(lab_slice_aug)
    
    return train_inputs, train_labels


def func_loadTrainingData(dataPath, patch_size, moduleIndex):

    train_inputs, train_labels = [], []

    img_C0_list = np.load(os.path.join(dataPath,'augmented_img_C0_list.npy'), allow_pickle=True)
    img_DE_list = np.load(os.path.join(dataPath,'augmented_img_DE_list.npy'), allow_pickle=True)
    img_T2_list = np.load(os.path.join(dataPath,'augmented_img_T2_list.npy'), allow_pickle=True)
    lab_list = np.load(os.path.join(dataPath,'augmented_lab_list.npy'), allow_pickle=True)

    numOfAug = len(img_C0_list)
    numOfCases = len(img_C0_list[0])
    
    for augIndex in range(numOfAug):
        for caseIndex in range(numOfCases):
            input_singlechannel_list = func_linearInputEncoder(img_C0_list, img_DE_list, img_T2_list, moduleIndex)
            train_inputs, train_labels = func_prepareTrainingData(patch_size, moduleIndex, inputCase=input_singlechannel_list[augIndex][caseIndex], train_inputs=train_inputs, labelCase=lab_list[augIndex][caseIndex], train_labels=train_labels)
    
    train_inputs = np.stack(train_inputs, axis=0)
    train_labels = np.stack(train_labels, axis=0)

    print('training input shape', train_inputs.shape)
    print('training label shape', train_labels.shape)

    return train_inputs, train_labels


def train(dataPath, batch_size, patch_size, num_epoch, save_frequency, validation_split, learning_rate):
    # create five U-nets for different labels

    for i in range(0, num_classes):  
        print('this is training with index', i)
        
        train_inputs_all, train_labels_all = func_loadTrainingData(dataPath, patch_size, moduleIndex=i)
        numOfTrainingSample = train_inputs_all.shape[0]

        train_inputs = train_inputs_all[:int(numOfTrainingSample * validation_split)]
        train_labels = train_labels_all[:int(numOfTrainingSample * validation_split)]
        print('number of training samples:', train_inputs.shape[0])

        val_inputs = train_inputs_all[int(numOfTrainingSample * validation_split):]
        val_labels = train_labels_all[int(numOfTrainingSample * validation_split):]
        print('number of validation samples:', val_inputs.shape[0])

        # visual debug
        # plt.figure()
        # plt.imshow(train_inputs[0,:,:,0])
        # plt.figure()
        # plt.imshow(train_labels[0,:,:])
        # plt.show()
        # plt.pause(0)
        
        model = UNet_model(patch_size, patch_size, learning_rate)

        model_root = os.path.join(os.getcwd(), 'Models', 'Model_UNet')
        history_root = os.path.join(os.getcwd(), 'History_UNet')
        model_name = f"model_UNet_{class_names[i]}_bs_{batch_size}_ps_{patch_size}_epoch_{num_epoch}_valid_split_{validation_split}_lr_{learning_rate}"
        model_dir = os.path.join(model_root, model_name)
        history_dir = history_root

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)

        save_model_name = os.path.join(model_dir, '{epoch:03d}.h5')
        save_callback = ModelCheckpoint(save_model_name, period=save_frequency)

        history = model.fit(train_inputs, train_labels, batch_size=batch_size, epochs=num_epoch, shuffle=True,
                                validation_data=(val_inputs, val_labels), callbacks=[save_callback])

        np.save(os.path.join(history_dir, model_name + '.npy'), history.history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default=os.path.join(os.getcwd(),'Data','Augmented_data','train_25'), help='input data folder after augmentation')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--ps', default=256, type=int, help='patch size')
    parser.add_argument('--epoch', default=250, type=int, help='number of epochs')
    parser.add_argument('--svfq', default=50, type=int, help='save period for UNet model')
    parser.add_argument('--vs', default=0.8, help='validation split from 0~1')
    parser.add_argument('--lr', default=1e-5, help='learning rate')
    args = parser.parse_args()

    print('Training configuration:',args)
    train(args.datapath, args.bs, args.ps, args.epoch, args.svfq, args.vs, args.lr)


    # jupyternotebook debug
    # train(os.path.join(os.getcwd(),'Data','Augmented_data','train_25_reduced'), 8, 256, 5, 1, 0.8, 1e-5)


# %%

# %%
from helper import func_imgNormalize, func_imgCrop, func_imgNormalize, func_linearInputEncoder, func_linearLabelEncoder
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import *
from network.UNet import dice_coef, dice_coef_loss
from keras.optimizers import Adam
from network.UNetplusplus.segmentation_models import Xnet
from keras.callbacks import ModelCheckpoint
import random
import argparse
from scipy.ndimage import rotate
from train_UNet import func_loadTrainingData

def train(dataPath, batch_size, patch_size, num_epoch, save_frequency, validation_split, learning_rate):
    # create five U-nets for different labels

    for i in range(3, num_classes):  
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
        
        # model = UNet_model(patch_size, patch_size, learning_rate)
        model = Xnet(backbone_name='vgg16', input_shape=(patch_size, patch_size, 1), encoder_weights=None, decoder_block_type='transpose')
        print(model.summary())
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

        model_root = os.path.join(os.getcwd(), 'Model', 'Model_UNetPP')
        history_root = os.path.join(os.getcwd(), 'History_UNetPP')
        model_name = f"model_UNetPP_{class_names[i]}_bs_{batch_size}_ps_{patch_size}_epoch_{num_epoch}_valid_split_{validation_split}_lr_{learning_rate}"
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
    parser.add_argument('--epoch', default=500, type=int, help='number of epochs')
    parser.add_argument('--svfq', default=50, type=int, help='save period for UNet model')
    parser.add_argument('--vs', default=0.8, help='validation split from 0~1')
    parser.add_argument('--lr', default=1e-5, help='learning rate')
    args = parser.parse_args()

    print('Training configuration:',args)
    train(args.datapath, args.bs, args.ps, args.epoch, args.svfq, args.vs, args.lr)


    # jupyternotebook debug
    # train(os.path.join(os.getcwd(),'Data','Augmented_data','train_25_reduced'), 8, 256, 5, 1, 0.8, 1e-5)


# %%

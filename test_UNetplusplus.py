# %%
from matplotlib.pyplot import axis
from config import *
from network.UNetplusplus.segmentation_models import Xnet
import os
import numpy as np
import tensorflow as tf
import nibabel as nib
import argparse
from keras.optimizers import Adam
from test_UNet import func_loadTestingData, func_saveTestPrediction
from network.UNet import dice_coef, dice_coef_loss

def test(dataPath, destPath, modelPath, moduleIndexList, batch_size, patch_size, num_epoch, validation_split, learning_rate):
    #debug_test_inputs, debug_test_names = [], []
    for moduleIndex in moduleIndexList:
        test_inputs, test_names = func_loadTestingData(dataPath, patch_size, moduleIndex)
        
        # debug_test_inputs.append(test_inputs), debug_test_names.append(test_names)
        model = Xnet(backbone_name='vgg16', input_shape=(patch_size, patch_size, 1), encoder_weights=None, decoder_block_type='transpose')
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

        model_name = f"model_UNetPP_{class_names[moduleIndex]}_bs_{batch_size}_ps_{patch_size}_epoch_{num_epoch}_valid_split_{validation_split}_lr_{learning_rate}"
        weights_path = os.path.join(modelPath, model_name, '{:03d}.h5'.format(num_epoch))
        model.load_weights(weights_path)

        test_predis = model.predict(test_inputs)
        predis_path = os.path.join(destPath, model_name)
        if not os.path.exists(predis_path):
            os.makedirs(predis_path)
        func_saveTestPrediction(predis_path, test_predis, test_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default=os.path.join(os.getcwd(),'Data','Original_data'), help='Test data path')
    parser.add_argument('--datadest', default=os.path.join(os.getcwd(),'Prediction', 'UNetPP'), help='Prediction path')
    parser.add_argument('--modelpath', default=os.path.join(os.getcwd(), 'Models', 'Model_UNetPP'))
    parser.add_argument('--moduleIndexList', default=[3, 4], help='model list for testing, 3:LV_ME, 4:LV_MS')
    parser.add_argument('--bs', default=8, type=int, help='batch size for testing model')
    parser.add_argument('--ps', default=256, type=int, help='patch size for testing model')
    parser.add_argument('--epoch', default=500, type=int, help='epoch for testing model')
    parser.add_argument('--vs', default=0.8, help='validation split for testing model')
    parser.add_argument('--lr', default=1e-5, help='learning rate for testing model')
    args = parser.parse_args()

    test(args.datapath, args.datadest, args.modelpath, args.moduleIndexList, args.bs, args.ps, args.epoch, args.vs, args.lr)

    # jupyternotebook debug
    # test(os.path.join(os.getcwd(),'Data','Original_data'), os.path.join(os.getcwd(),'Prediction', 'UNetPP'), os.path.join(os.getcwd(),'Models','Model_UNetPP'), [3,4], 8, 256, 5, 0.8, 1e-5)

#%%


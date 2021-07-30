import os
import sys
import random
import math
import re
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import h5py

#from mrcnnv2.config import Config
#from mrcnnv2 import utils
#import mrcnnv2.model as modellib
#from mrcnnv2 import visualize
#from mrcnnv2.model import log

from network.maskrcnn.mrcnn.config import Config
from network.maskrcnn.mrcnn import utils
import network.maskrcnn.mrcnn.model as modellib
from network.maskrcnn.mrcnn import visualize
from network.maskrcnn.mrcnn.model import log

from helper import func_createLabelToMask, func_imgCrop

##########################################
# Some hyperparameters
##########################################

crop_size = 256 # Image crop size

class MyoConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "MyoPS"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + LV_MS or LV_ME

    # backbone model
    BACKBONE = "resnet50"

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 256
    # IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 16

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 20


class MyoDataset(utils.Dataset):   
    
    def load_myops(self, path, subset, val_split, mode):
        # add classes
        assert mode in ['LV_ME', 'LV_MS']
        self.add_class('myo', 1, mode)
        #self.add_class('myo', 1, "LV_MS")
        assert subset in ["train", "val"]

        # add input images not the masks with two modes
        path_load = os.path.join(path, mode, 'input')

        n_data = len(os.listdir(path_load))
        
        if subset == 'train':
            print('total number of training samples',int(n_data*val_split))
            input_names = os.listdir(path_load)[:int(n_data*val_split)]
        if subset == 'val':
            print('total number of validation samples',int(n_data-n_data*val_split))
            input_names = os.listdir(path_load)[int(n_data*val_split):]

        
        # add the input infos
        for input_name in input_names:
            input_ = np.load(os.path.join(path_load, input_name))
            #print(input_.shape)
            height, width, _ = input_.shape
            self.add_image('myo',image_id=input_name, width=width, height=height,path=path_load)
            


        #with h5py.File(fname_h5, 'r') as f:
        #    if subset == 'train':
        #        list_f = list(f)[:ndata]
        #    else:
        #        list_f = list(f)[ndata:]
        #        
        #    for pid in list_f:
        #        height, width = f[pid]['im'].shape
        #        self.add_image('thyroid', image_id=pid, width=width, height=height, pid=pid, path=fname_h5)
                
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "myo":
            return info["myo"]
        else:
            super(self.__class__).image_reference(self, image_id)     

#     def im_normalize(self, im, range_=[1.0,99.0]):
#         im_min, im_max = np.percentile(im,range_)
#         return np.clip(np.array((im-im_min)/(im_max-im_min), dtype=np.float32), 0.0, 1.0)               
            
    def load_image(self, image_id):
        info = self.image_info[image_id]
        #print(image_id)
        input_ = np.load(os.path.join(info['path'],info['id']))  
        input_ = np.stack((input_,)*3, axis=2)
        return input_[:,:,:,0]       
            
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path_mask = info['path'].replace('input','label')
        #print('loading mask',self.class_names[1])
        label = np.load(os.path.join(path_mask,info['id']))
        label = func_imgCrop(label,info['height'])
        count = len(self.class_names)
        #print('number of class (including the BG):',count)
        mask = np.zeros((info['height'],info['width'],count),dtype=np.uint8)
        class_ids = []
        for i in range(count):
            mask[:,:,i] = func_createLabelToMask(label,self.class_names[i])
            class_ids.append(i)
        
        class_ids = np.array(class_ids,dtype=np.int32)
        return mask, class_ids


# %%
from re import T
from network.maskrcnn.myops_mrcnn import *
import os
from network.maskrcnn.mrcnn import visualize
from network.maskrcnn.mrcnn.config import Config
import argparse
from train_UNet import func_loadTrainingData, func_imgPairRandomRotation
from helper import func_saveTrainingDataIntoSlices, func_linearInputEncoder, func_linearLabelEncoder

ISTRAIN = True

COCO_MODEL_PATH = os.path.join(os.getcwd(),'mask_rcnn_coco.h5')

def func_prepareTrainingData_mrcnn(patch_size, inputCase, train_inputs, labelCase, train_labels):
    numofLayer = inputCase.shape[2]
    for layerIndex in range(numofLayer):
        img_slice = func_imgCrop(inputCase[:,:,layerIndex], patch_size)
        img_slice = np.expand_dims(img_slice, axis=2)
        train_inputs.append(img_slice)

        lab_slice = func_imgCrop(labelCase[:,:,layerIndex], patch_size)
        train_labels.append(lab_slice)

        img_slice_aug, lab_slice_aug = func_imgPairRandomRotation(img_slice, lab_slice)
        train_inputs.append(img_slice_aug)
        train_labels.append(lab_slice_aug)
    
    return train_inputs, train_labels


def func_loadTrainingData_mrcnn(dataPath, patch_size, moduleIndex):

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
            train_inputs, train_labels = func_prepareTrainingData_mrcnn(patch_size, inputCase=input_singlechannel_list[augIndex][caseIndex], train_inputs=train_inputs, labelCase=lab_list[augIndex][caseIndex], train_labels=train_labels)
    
    train_inputs = np.stack(train_inputs, axis=0)
    train_labels = np.stack(train_labels, axis=0)

    print('training input shape', train_inputs.shape)
    print('training label shape', train_labels.shape)

    return train_inputs, train_labels


def train(mode, datapath, modeldir, num_epoch, patch_size, validation_split, learning_rate):
    #path_data = os.path.join(os.path.join(root_,'Data'),'Training_data_priori')
    model_name = f"model_MaskRCNN_{mode}_ps_{patch_size}_epoch_{num_epoch}_vs_{validation_split}"
    MODEL_DIR = os.path.join(modeldir, model_name)
    NEPOCHS = num_epoch

    moduleIndex = 3 if mode == 'LV_ME' else 4
    train_inputs, train_labels = func_loadTrainingData_mrcnn(os.path.join(os.getcwd(), 'Data', 'Augmented_data','train_25'), patch_size, moduleIndex)
    destPath = os.path.join(os.path.join(os.getcwd(), 'Data', 'Augmented_data', 'maskrcnn'))
    func_saveTrainingDataIntoSlices(destPath, train_inputs, train_labels, mode)

    dataset_train = MyoDataset()
    dataset_train.load_myops(datapath, 'train', val_split=validation_split, mode=mode)
    dataset_train.prepare()
    
    dataset_val = MyoDataset()
    dataset_val.load_myops(datapath, 'val', val_split=validation_split, mode=mode)
    dataset_val.prepare()

    # image_ids = np.random.choice(dataset_val.image_ids, 4)

    # for image_id in image_ids:
    #     print (image_id, dataset_train.image_info[image_id]['id'])
    #     im = dataset_train.load_image(image_id)
    #     mask, class_ids = dataset_train.load_mask(image_id)
    #     visualize.display_top_masks(im, mask, class_ids, dataset_train.class_names)
    

    config = MyoConfig()
    config.display()
    model = modellib.MaskRCNN(mode="training",config=config,model_dir=MODEL_DIR)
    

    if ISTRAIN:
        # Which weights to start with?
        init_with = "coco"  # imagenet, coco, or last

        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            model.load_weights(model.find_last(), by_name=True)
    
    if ISTRAIN:
        model.train(dataset_train, dataset_val, learning_rate=learning_rate, epochs=NEPOCHS, layers='heads')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help='LV_ME or LV_MS')
    parser.add_argument('--datapath', default=os.path.join(os.getcwd(), 'Data', 'Augmented_data', 'maskrcnn'), help='input data folder after augmentation')
    parser.add_argument('--modeldir', default=os.path.join(os.getcwd(), 'Models', 'Model_MaskRCNN'), help='path to save MaskRCNN model')
    parser.add_argument('--epoch', default=500, type=int, help='number of epoch')
    parser.add_argument('--ps', default=256, type=int, help='patch size')
    parser.add_argument('--vs', default=0.8, help='validation split')
    parser.add_argument('--lr', default=0.001, help='learning rate')
    args = parser.parse_args()
    
    print('Training configuration:',args)
    train(args.mode, args.datapath, args.modeldir, args.epoch, args.ps, args.vs, args.lr)

    # jupyternotebook debug
    # train('LV_MS', os.path.join(os.getcwd(), 'Data', 'Augmented_data', 'maskrcnn'), os.path.join(os.getcwd(), 'Models', 'Model_MaskRCNN'), 5, 256, 0.8, 0.001)
# %%
    


# %%

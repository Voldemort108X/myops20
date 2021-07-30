#from myops_priori_LV_ME import *
import numpy as np
from network.maskrcnn.myops_mrcnn import *
import os
from network.maskrcnn.mrcnn import visualize
from network.maskrcnn.mrcnn.config import Config
from test_UNet import func_createTestImageList
from helper import func_linearInputEncoder
import argparse

class InferenceConfig(MyoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MyoDataset_test(utils.Dataset):   
    
    def load_myops(self, path, subset, mode):
        # add classes
        self.add_class('myo', 1, mode)
        assert subset in ["test"]

        # add input images not the masks with two modes
        path_load = os.path.join(path, mode, 'input')

        if subset == 'test':
            input_names = os.listdir(path_load)

        # add the input infos
        for input_name in input_names:
            input_ = np.load(os.path.join(path_load,input_name))
            height, width, _ = input_.shape
            self.add_image('myo',image_id=input_name,width=width,height=height,path=path_load)
            
        
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "myo":
            return info["myo"]
        else:
            super(self.__class__).image_reference(self, image_id)                   
            
    def load_image(self, image_id):
        info = self.image_info[image_id]
        #print(image_id)
        input_ = np.load(os.path.join(info['path'],info['id']))  
        input_ = np.stack((input_,)*3, axis=2)
        return input_[:,:,:,0]          
            

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def func_prepareTestingData(patch_size, inputCase, test_inputs, inputCaseName, test_names):
    numOfLayer = inputCase.shape[2]
    for layerIndex in range(numOfLayer):
        img_slice = func_imgCrop(inputCase[:,:,layerIndex], patch_size)
        img_slice = np.expand_dims(img_slice, axis=2)
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

def func_saveTestingDataIntoSlices(destPath, test_inputs, test_names, mode):
    numOfData = len(test_inputs)
    print('number of test sample:', numOfData)
    destPath_input = os.path.join(destPath, mode, 'input')
    if not os.path.exists(destPath_input):
        os.makedirs(destPath_input)

    for i in range(numOfData):
        np.save(os.path.join(destPath_input, test_names[i].replace('gd','input')+ '.npy'), test_inputs[i])

def test(mode, dataPath, destPath, modeldir, patch_size, num_epoch, validation_split):

    model_name = f"model_MaskRCNN_{mode}_ps_{patch_size}_epoch_{num_epoch}_vs_{validation_split}"
    model_root_path = os.path.join(modeldir, model_name)
    inference_config = InferenceConfig()

    # recreate the model in the reference mode
    model = modellib.MaskRCNN(mode="inference",config=inference_config,model_dir=model_root_path)

    model_path = model.find_last()

    model.load_weights(model_path, by_name=True)
    
    moduleIndex = 3 if mode == 'LV_ME' else 4
    test_inputs, test_names = func_loadTestingData(os.path.join(os.getcwd(), 'Data', 'Original_data'), patch_size, moduleIndex)
    print(test_names)

    destPath_input = dataPath
    func_saveTestingDataIntoSlices(destPath_input, test_inputs, test_names, mode)

    dataset_test = MyoDataset_test()
    dataset_test.load_myops(dataPath, 'test', mode)
    dataset_test.prepare()
    

    for image_id in dataset_test.image_ids:
        test_ = np.load(os.path.join(dataPath, mode, 'input', dataset_test.image_info[image_id]['id']))
        img_size = test_.shape[0]
        predi = np.zeros((img_size,img_size,1))
        results = model.detect([test_],verbose=0) # verbose=1 indicates printing log info
        r = results[0]
        num_instance = r['masks'].shape[-1]
        if num_instance !=0:
            for inst in range(num_instance):
                print('detected',num_instance,'regions')
                print('shape of masks',r['masks'][:,:,inst].shape)
                class_id = r['class_ids'][inst]
                class_name = dataset_test.class_names[class_id]
                if class_name == 'LV_ME':
                    print('saving LV_ME')
                    predi = np.expand_dims(r['masks'][:,:,inst].astype(np.int),axis=2) + predi
                if class_name == 'LV_MS':
                    print('saving LV_MS')
                    predi = np.expand_dims(r['masks'][:,:,inst].astype(np.int),axis=2) + predi

        path_predi = os.path.join(destPath, model_name)

        if not os.path.exists(path_predi):
            os.makedirs(path_predi)

        np.save(os.path.join(path_predi, dataset_test.image_info[image_id]['id'].replace('input','predi')), predi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help='LV_ME or LV_MS')
    parser.add_argument('--datapath', default=os.path.join(os.getcwd(), 'Data', 'Test_data', 'maskrcnn'), help='input data folder after augmentation')
    parser.add_argument('--destpath', default=os.path.join(os.getcwd(),'Prediction','MaskRCNN'), help='output data folder aftere augmentation')
    parser.add_argument('--modeldir', default=os.path.join(os.getcwd(), 'Models', 'Model_MaskRCNN'))
    parser.add_argument('--ps', default=256, type=int, help='patch size')
    parser.add_argument('--epoch', default=500, type=int, help='number of epoch')
    parser.add_argument('--vs', default=0.8, help='validation split')
    args = parser.parse_args()

    test(args.mode, args.datapath, args.destpath, args.modeldir, args.ps, args.epoch, args.vs)

    # jupyternotebook debug
    # test('LV_MS',os.path.join(os.getcwd(), 'Data', 'Test_data', 'maskrcnn'),os.path.join(os.getcwd(),'Prediction','MaskRCNN'),os.path.join(os.getcwd(), 'Models', 'Model_MaskRCNN'), 256, 5, 0.8)

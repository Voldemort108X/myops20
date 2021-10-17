# MyoPS 2020: Fully Automated Deep Learning-based Segmentation of Normal, Infarcted and Edema Regions from Multiple Cardiac MRI Sequences
## [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-65651-5_8) [[Challenge website]](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/myops20/index.html)

## Overview
![Alt](asset/MyoPS20.png "Overiew")

## Bibtex
```
@inproceedings{zhang2020fully,
  title={Fully automated deep learning based segmentation of normal, infarcted and edema regions from multiple cardiac MRI sequences},
  author={Zhang, Xiaoran and Noga, Michelle and Punithakumar, Kumaradevan},
  booktitle={Myocardial Pathology Segmentation Combining Multi-Sequence CMR Challenge},
  pages={82--91},
  year={2020},
  organization={Springer}
}
```
## Dataset
Please refer to challenge website [[link]](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/myops20/index.html) for dataset access. The dataset contains three folders: train25, train25_myops_gd, test20.

## Environment
1. UNet: 
```
conda env create -f myops_unet.yml
```
2. Mask-RCNN and UNet++:
```
conda env create -f myops_mrcnn_unetpp.yml
```
## Default directory structure

    ├── Data                   
    |   ├── Original_data       # Place the downloaded dataset here
    |   |   ├── train25
    |   |   ├── train25_myops_gd
    |   |   ├── test20
    ├── mask_rcnn_coco.h5       # Downloaded pre-trained mask_rcnn weights
    ├── config.py
    ├── data_creator.py
    ├── ...


## Setup
1. Data creator including random warping augmentation
```
python data_creator.py 
```

2. Train networks
    - Train UNet for LV_BP, RV_BP, LV_NM, LV_ME, LV_MS blocks
    ```
    python train_UNet.py
    ```
    - Train Mask-RCNN for LV_ME and LV_MS blocks
        - Download pretrained mask_rcnn_coco.h5 at [[here]](https://github.com/matterport/Mask_RCNN/releases) and place it in the current folder.
        - Train Mask-RCNN for LV_ME block
        ```
        python train_MaskRCNN.py --mode 'LV_ME'
        ```
        - Train Mask-RCNN for LV_MS block
        ```
        python train_MaskRCNN.py --mode 'LV_MS'
        ```
    - Train UNet++ for LV_ME and LV_MS blocks
    ```
    python train_UNetplusplus.py
    ```
3. Test networks:
    - Test UNet
    ```
    python test_UNet.py
    ```
    - Test Mask-RCNN
        - Test LV_ME block
        ```
        python test_MaskRCNN.py --mode 'LV_ME'
        ```
        - Test LV_MS block
        ```
        python test_MaskRCNN.py --mode 'LV_MS'
        ```
    - Test UNet++
    ```
    python test_UNetplusplus.py
    ```

4. Post-processing and linear decoder:
```
python post_processing.py
```
## Acknowledgement
1. Please cite the official Mask-RCNN and UNet++ implementations if you use them:
    - Mask-RCNN: https://github.com/matterport/Mask_RCNN
    - UNet++: https://github.com/MrGiovanni/UNetPlusPlus
2. The authors would wish to acknowledge Compute Canada for providing the computation resource.

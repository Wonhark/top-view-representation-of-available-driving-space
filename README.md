# top-view-representation-of-available-driving-space

## Introduction
Using deeplearning, segmentation and depth estimation can be done on image. When driving, knowing where the obstacles are is very important. 
We can segment obstacles and represent only those segmented obstacles in top-view image. By this algorithm, driver can tell drivable space, such as road with no obstacles.
In this repository, we segmented out only the road and took any other things as obstacles. By using limitation on height(e.g. 3m, where car is likely to hit), we can take things that are above the height of the car out of account, e.g. sky.

## Inference
Inference is easy: just run main.ipynb!! 
Since pre-trained models are not included in this repository, pre-trained models need to be downloaded. links are at the down below.
So, here are the procedure:

1. Download pre-trained models from github pages of "Monodepth2" and "DeepLabV3Plus".

2. get images (data) in data/image_folder and modify the image_path in main.ipynb.

3. run main.ipynb

Used scripts are ./main.ipynb, ./top_view_represent.py, ./DeepLabV3Plus_Pytorch_master/segmentation_predict.py, ./monodepth2_master/depth_estimation.py

Rest python file are the ones in reference github pages, and some don't belong to this algorithm.


This work has used pre-trained model of "Monodepth2" as depth estimation neural network and pre-trained model of "DeepLabV3Plus" as segmentation neural network.

Pre-trained model:

1. Monodepth2: mono+stereo_640x192

it's available in Monodepth2's github.

2. DeepLabV3Plus: best_deeplabv3plus_mobilenet_cityscapes_os16.pth

it's available in DeepLabV3Plus's github - dropbox.


## Reference
1. Monodepth2
@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
year = {2019}
}

https://github.com/nianticlabs/monodepth2


2. DeepLabV3Plus
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}

https://github.com/VainF/DeepLabV3Plus-Pytorch


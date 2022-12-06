# top-view-representation-of-available-driving-space

Using deeplearning, segmentation and depth estimation can be done on image. When driving, knowing where the obstacles are is very important. 
We can segment obstacles and represent only those segmented obstacles in top-view image. By this algorithm, driver can tell drivable space, such as road with no obstacles.
In this repository, we segmented out only the road and took any other things as obstacles. By using limitation on height(e.g. 3m, where car is likely to hit), we can take things that are above the height of the car out of account, e.g. sky.

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

https://github.com/VainF/DeepLabV3Plus-Pytorch


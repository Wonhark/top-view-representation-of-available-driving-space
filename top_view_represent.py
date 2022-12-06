##### From depth image, calculate the camera coordinate(x,y,z) 
##### and represent segmentation image in (x,z) as top-view.

import torch
import numpy as np
from numpy import loadtxt
from PIL import Image


def top_view_representation(depth_np, seg_np, image_path, K_path, top_view_size=(1000,1000), depth_scale=0.03):
    # get numpy depth array
    H, W = depth_np.shape
    
    # get numpy segmentation array. It's the result from segmentation NN model.
    
    pil_seg = Image.fromarray(np.uint8(seg_np)).convert('L').resize((W,H), resample=Image.Resampling(0))
    seg_np = np.array(pil_seg)
    
    # get original image png file. 
    # Since camera intrinsic is based on original image pixel size, it's needed.
    img = Image.open(image_path).convert('RGB')
    H_img,W_img,_ = np.array(img, dtype=np.float32).shape
    
    # get intrinsic K txt matrix.
    K = loadtxt(K_path)
    cx, cy = K[0,2], K[1,2]
    fx, fy = K[0,0], K[1,1]
    
    # define new numpy array: top-view numpy array, which will be converted to image later.
    H_tv, W_tv = top_view_size
    seg_top_view = np.zeros([H_tv, W_tv])
    x_scale, y_scale = W_img/W, H_img/H
    
    # using camera intrinsics, calculate global(camera) coordinate (x,y,z).
    for i in range(H):
        for j in range(W):
            d = depth_np[i,j]
            s = seg_np[i,j]
            z = d / depth_scale
            x = (j* x_scale - cx)*z / fx 
            y = (i* y_scale - cy)*z / fy 

            x = int(x) + W_tv//2
            z = H_tv-int(z)-1

            if x<W_tv and x>=0 and z<H_tv and z>=0 and y<=3/ depth_scale:
                seg_top_view[z, x] = s
    
    pil_seg_img = Image.fromarray(np.uint8(seg_top_view*255)).convert('L')
    return pil_seg_img
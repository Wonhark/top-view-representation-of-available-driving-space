from torch.utils.data import dataset
from tqdm import tqdm
from . import network
from . import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from .datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from .metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

class Configuration():
    def __init__(self):
        self.dataset = "cityscapes"
        self.input = "image1.jpg"
        self.model = "deeplabv3plus_mobilenet"
        self.ckpt = "best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
        self.save_val_results_to = "segmentation_results"
        self.gpu_id = "0"
        self.num_classes = 19
        self.output_stride = 16
        self.separable_conv = False
        self.crop_val = False
        self.crop_size = 513

def DeepLabV3Plus_segmentation(image_path, model):
    opts = Configuration()
    opts.input = image_path

    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print("Device: %s" % device)

    # Setup dataloader
    image_files = [image_path]
    
    model.to(device)


    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = T.Compose([
                T.Resize(opts.crop_size),
                T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in image_files:
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            
            pred = model(img).max(1)[1].cpu().numpy()[0] # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, img_name+'.png'))

    # numpy array conversion
    img = Image.open(os.path.join(opts.save_val_results_to, img_name+'.png'), 'r')
    w, h = img.size
    pix = list(img.getdata())
    pix = [pix[n:n+w] for n in range(0, w*h, w)]
    newarr = [[1 for y in range(len(pix[0]))] for x in range(len(pix))]

    # only take road as 1, else 0
    for x in range(len(pix)):
        for y in range(len(pix[0])):
            if pix[x][y] == (128, 64, 128):
                newarr[x][y] = 0

    #np.save("test3", newarr)
    #new = np.load("test3.npy")

    return newarr
"""
if __name__ == '__main__':
    main()
"""
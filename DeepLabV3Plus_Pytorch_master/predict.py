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
        self.save_val_results_to = "test_results"
        self.gpu_id = "0"
        self.num_classes = 19
        self.output_stride = 16
        self.separable_conv = False
        self.crop_val = False
        self.crop_size = 513

def DeepLabV3Plus_segmentation():
    opts = Configuration()

    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
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
        for img_path in tqdm(image_files):
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
    img = Image.open('./test_results/image1.png', 'r')
    w, h = img.size
    pix = list(img.getdata())
    pix = [pix[n:n+w] for n in range(0, w*h, w)]
    newarr = [[1 for y in range(len(pix[0]))] for x in range(len(pix))]

    for x in range(len(pix)):
        for y in range(len(pix[0])):
            if pix[x][y] == (128, 64, 128):
                newarr[x][y] = 0

    np.save("test3", newarr)
    new = np.load("test3.npy")

    return new

if __name__ == '__main__':
    main()

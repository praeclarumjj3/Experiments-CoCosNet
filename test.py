# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import data
import numpy as np
from util.util import masktorgb
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
import warnings
from termcolor import colored
warnings.filterwarnings("ignore")

print(colored('OUTPUT STARTS', 'cyan'))

opt = TestOptions().parse()
   
torch.manual_seed(0)
dataloader = data.create_dataloader(opt)
dataloader.dataset[0]

model = Pix2PixModel(opt)
model.eval()

save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output')

# test
for i, data_i in enumerate(dataloader):
    print('{} / {}'.format(i, len(dataloader)))
    if i * opt.batchSize >= opt.how_many:
        break
    imgs_num = data_i['label'].shape[0]
    
    print(data_i.keys())
    out = model(data_i, mode='inference')
    
    # Uncomment to save single images
    # root = save_root + '/test_per_img/'
    # if not os.path.exists(root + opt.name):
    #     os.makedirs(root + opt.name)
    
    # imgs = out['fake_image'].data.cpu()
    # label = masktorgb(data_i['label'].cpu().numpy())
    # label = torch.from_numpy(label).float() / 128 - 1
    
    # try:
    #     for i in range(imgs.shape[0]):
    #         name = os.path.basename(data_i['path'][i])
    #         imgs_save = torch.cat((label[i:i+1].cpu(), data_i['ref'][i:i+1].cpu(), imgs[i:i+1]), 0)
    #         imgs_save = (imgs_save + 1) / 2
    #         vutils.save_image(imgs_save, root + opt.name + '/' + name,  
    #                 nrow=3, padding=0, normalize=False)
    # except OSError as err:
    #     print(err)

    ## save array of images
    if not os.path.exists(save_root + '/test/' + opt.name):
        os.makedirs(save_root + '/test/' + opt.name)

    label = masktorgb(data_i['label'].cpu().numpy())
    label = torch.from_numpy(label).float() / 128 - 1
    imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), out['fake_image'].data.cpu()), 0)
    try:
        imgs = (imgs + 1) / 2
        vutils.save_image(imgs, save_root + '/test/' + opt.name + '/' + str('out') + '.png',  
                nrow=imgs_num, padding=0, normalize=False)
    except OSError as err:
        print(err)
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import models.networks as networks
import util.util as util


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        self.alpha = 1

        self.net = torch.nn.ModuleDict(self.initialize_networks(opt))

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode, GforD=None, alpha=1):
        input_semantics, self_ref, ref_image, ref_semantics = self.preprocess_input(data, )

        self.alpha = alpha
        if mode == 'inference':
            out = {}
            with torch.no_grad():
                out = self.inference(input_semantics, 
                        ref_semantics=ref_semantics, ref_image=ref_image, self_ref=self_ref)
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            return out
        else:
            raise ValueError("|mode| is invalid")

    def initialize_networks(self, opt):
        net = {}
        net['netG'] = networks.define_G(opt) # Get SPADEGenerator
        net['netCorr'] = networks.define_Corr(opt) # Get NoVGGCorrespondence

        if not opt.isTrain or opt.continue_train:
            net['netG'] = util.load_network(net['netG'], 'G', opt.which_epoch, opt)
            net['netCorr'] = util.load_network(net['netCorr'], 'Corr', opt.which_epoch, opt)
        return net

    def preprocess_input(self, data):
        '''
        data has 6 keys: ['label', 'image', 'path', 'self_ref', 'ref', 'label_ref']
        label: .png maps in the val folder
        image: .jpg images in the val folder
        path: .jpg files path in val folder
        self_ref: a tensor with all zeros
        ref: .jpg image for ref from the training folder
        label_ref: .png file from train
        '''
        # move to GPU and change data types
        if self.opt.dataset_mode != 'deepfashion':
            data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()
            data['ref'] = data['ref'].cuda()
            data['label_ref'] = data['label_ref'].cuda()
            if self.opt.dataset_mode != 'deepfashion':
                data['label_ref'] = data['label_ref'].long()
            data['self_ref'] = data['self_ref'].cuda()

        # create one-hot label map
        if self.opt.dataset_mode != 'celebahqedge' and self.opt.dataset_mode != 'deepfashion':
            label_map = data['label']
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0)
        
            label_map = data['label_ref']
            label_ref = self.FloatTensor(bs, nc, h, w).zero_()
            ref_semantics = label_ref.scatter_(1, label_map, 1.0)
        return input_semantics, data['self_ref'], data['ref'], ref_semantics

    def inference(self, input_semantics, ref_semantics=None, ref_image=None, self_ref=None):
        generate_out = {}
        coor_out = self.net['netCorr'](ref_image, None, input_semantics, ref_semantics, alpha=self.alpha)
        
        if self.opt.CBN_intype == 'mask':
            CBN_in = input_semantics
        elif self.opt.CBN_intype == 'warp':
            CBN_in = coor_out['warp_out']
        elif self.opt.CBN_intype == 'warp_mask':
            CBN_in = torch.cat((coor_out['warp_out'], input_semantics), dim=1)

        generate_out['fake_image'] = self.net['netG'](input_semantics, warp_out=CBN_in)
        generate_out = {**generate_out, **coor_out}
        return generate_out

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

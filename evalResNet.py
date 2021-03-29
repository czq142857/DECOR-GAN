import os
import time
import math
import random
import numpy as np
import h5py
import cv2
from scipy.ndimage.filters import gaussian_filter

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

import cutils
import binvox_rw_faster as binvox_rw

from utils import *


#define a simple ResNet for faster training
#input 64 (cropped from 256)

class resnet_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(resnet_block, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        if self.dim_in == self.dim_out:
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=1, bias=False)
            self.bn_1 = nn.InstanceNorm2d(self.dim_out)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
            self.bn_2 = nn.InstanceNorm2d(self.dim_out)
        else:
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=2, padding=1, bias=False)
            self.bn_1 = nn.InstanceNorm2d(self.dim_out)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
            self.bn_2 = nn.InstanceNorm2d(self.dim_out)
            self.conv_s = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=2, padding=0, bias=False)
            self.bn_s = nn.InstanceNorm2d(self.dim_out)

    def forward(self, input):
        if self.dim_in == self.dim_out:
            output = self.bn_1(self.conv_1(input))
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
            output = self.bn_2(self.conv_2(output))
            output = output+input
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
        else:
            output = self.bn_1(self.conv_1(input))
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
            output = self.bn_2(self.conv_2(output))
            input_ = self.bn_s(self.conv_s(input))
            output = output+input_
            output = F.leaky_relu(output, negative_slope=0.02, inplace=True)
        return output

class img_encoder(nn.Module):
    def __init__(self, img_ef_dim, z_dim):
        super(img_encoder, self).__init__()
        self.img_ef_dim = img_ef_dim
        self.z_dim = z_dim
        self.conv_0 = nn.Conv2d(1, self.img_ef_dim, 7, stride=2, padding=3, bias=False)
        self.bn_0 = nn.InstanceNorm2d(self.img_ef_dim)
        self.res_1 = resnet_block(self.img_ef_dim, self.img_ef_dim)
        self.res_2 = resnet_block(self.img_ef_dim, self.img_ef_dim)
        self.res_3 = resnet_block(self.img_ef_dim, self.img_ef_dim*2)
        self.res_4 = resnet_block(self.img_ef_dim*2, self.img_ef_dim*2)
        self.res_5 = resnet_block(self.img_ef_dim*2, self.img_ef_dim*4)
        self.res_6 = resnet_block(self.img_ef_dim*4, self.img_ef_dim*4)
        self.conv_9 = nn.Conv2d(self.img_ef_dim*4, self.img_ef_dim*4, 4, stride=2, padding=1, bias=False)
        self.bn_9 = nn.InstanceNorm2d(self.img_ef_dim*4)
        self.conv_10 = nn.Conv2d(self.img_ef_dim*4, self.z_dim, 4, stride=1, padding=0, bias=True)

    def forward(self, view):
        layer_0 = self.bn_0(self.conv_0(1-view))
        layer_0 = F.leaky_relu(layer_0, negative_slope=0.02, inplace=True)

        layer_1 = self.res_1(layer_0)
        layer_2 = self.res_2(layer_1)
        
        layer_3 = self.res_3(layer_2)
        layer_4 = self.res_4(layer_3)
        
        layer_5 = self.res_5(layer_4)
        layer_6 = self.res_6(layer_5)
        
        layer_9 = self.bn_9(self.conv_9(layer_6))
        layer_9 = F.leaky_relu(layer_9, negative_slope=0.02, inplace=True)
        
        layer_10 = self.conv_10(layer_9)
        layer_10 = layer_10.view(-1)

        return layer_10


def eval_Cls_score(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    output_dir = "eval_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fake_dir = "render_fake_for_eval"
    if not os.path.exists(fake_dir):
        print("ERROR: fake_dir does not exist! "+fake_dir)
        exit(-1)

    real_dir = "render_real_for_eval"
    if not os.path.exists(real_dir):
        print("ERROR: real_dir does not exist! "+real_dir)
        exit(-1)

    sample_num_views = 24
    view_size = 256
    crop_size = 64
    if config.output_size==128:
        img_ef_dim = 8
        batch_size = 1024
    elif config.output_size==256:
        img_ef_dim = 16
        batch_size = 1024


    #load datasets
    print("loading images")

    data_real_names = os.listdir(real_dir)
    data_real_len = len(data_real_names)
    print("real images:", data_real_len)
    data_real_pixels = np.zeros([data_real_len,1,view_size,view_size],np.float32)
    for i in range(data_real_len):
        img = cv2.imread(real_dir+"/"+data_real_names[i], cv2.IMREAD_UNCHANGED)
        if config.output_size==128:
            data_real_pixels[i,0] = gaussian_filter(img.astype(np.float32), sigma=2) #remove small local noise
        elif config.output_size==256:
            data_real_pixels[i,0] = gaussian_filter(img.astype(np.float32), sigma=1) #remove small local noise

    data_fake_names = os.listdir(fake_dir)
    data_fake_len = len(data_fake_names)
    print("fake images:", data_fake_len)
    data_fake_pixels = np.zeros([data_fake_len,1,view_size,view_size],np.float32)
    for i in range(data_fake_len):
        img = cv2.imread(fake_dir+"/"+data_fake_names[i], cv2.IMREAD_UNCHANGED)
        if config.output_size==128:
            data_fake_pixels[i,0] = gaussian_filter(img.astype(np.float32), sigma=2) #remove small local noise
        elif config.output_size==256:
            data_fake_pixels[i,0] = gaussian_filter(img.astype(np.float32), sigma=1) #remove small local noise


    # define network

    resnet = img_encoder(img_ef_dim, 1)
    resnet.to(device)
    optimizer = torch.optim.Adam(resnet.parameters())


    # training

    shape_num = min(data_real_len,data_fake_len)
    real_index_list = np.arange(data_real_len)
    fake_index_list = np.arange(data_fake_len)


    print("\n\n----------net summary----------")
    print("real samples   ", data_real_len)
    print("fake samples   ", data_fake_len)
    print("-------------------------------\n\n")


    start_time = time.time()
    training_epoch = 200
    batch_num = int(shape_num/batch_size)

    ones = torch.ones([batch_size]).to(device)
    zeros = torch.zeros([batch_size]).to(device)

    resnet.train()
    for epoch in range(training_epoch):
        np.random.shuffle(real_index_list)
        np.random.shuffle(fake_index_list)
        avg_real_score = 0
        avg_fake_score = 0
        avg_num = 0
        for idx in range(batch_num):
            real_batch_ids = real_index_list[idx*batch_size:(idx+1)*batch_size]
            fake_batch_ids = fake_index_list[idx*batch_size:(idx+1)*batch_size]

            real_crop_x = np.random.randint(view_size-crop_size)
            real_crop_y = np.random.randint(view_size-crop_size)
            fake_crop_x = np.random.randint(view_size-crop_size)
            fake_crop_y = np.random.randint(view_size-crop_size)

            batch_real_ = data_real_pixels[real_batch_ids,:,real_crop_y:real_crop_y+crop_size,real_crop_x:real_crop_x+crop_size]
            batch_fake_ = data_fake_pixels[fake_batch_ids,:,fake_crop_y:fake_crop_y+crop_size,fake_crop_x:fake_crop_x+crop_size]

            batch_real = torch.from_numpy(batch_real_).to(device)/255.0
            batch_fake = torch.from_numpy(batch_fake_).to(device)/255.0

            resnet.zero_grad()
            logits_real = resnet(batch_real)
            logits_fake = resnet(batch_fake)

            loss_real = F.binary_cross_entropy_with_logits(logits_real, ones)
            loss_fake = F.binary_cross_entropy_with_logits(logits_fake, zeros)
            loss = loss_real + loss_fake

            loss.backward()
            optimizer.step()

            real_score = torch.mean((logits_real>0).float())
            fake_score = torch.mean((logits_fake<0).float())

            avg_real_score += real_score.item()
            avg_fake_score += fake_score.item()
            avg_num += 1

        print("Epoch: [%2d/%2d] time: %4.4f, real_score: %.8f, fake_score: %.8f" % (epoch, training_epoch, time.time() - start_time, avg_real_score/avg_num, avg_fake_score/avg_num))


    # testing
    
    resnet.eval()

    avg_real_score = 0
    avg_fake_score = 0
    avg_num = 0
    for xxx in range(10):
        np.random.shuffle(real_index_list)
        np.random.shuffle(fake_index_list)
        for idx in range(batch_num):
            real_batch_ids = real_index_list[idx*batch_size:(idx+1)*batch_size]
            fake_batch_ids = fake_index_list[idx*batch_size:(idx+1)*batch_size]

            real_crop_x = np.random.randint(view_size-crop_size)
            real_crop_y = np.random.randint(view_size-crop_size)
            fake_crop_x = np.random.randint(view_size-crop_size)
            fake_crop_y = np.random.randint(view_size-crop_size)

            batch_real_ = data_real_pixels[real_batch_ids,:,real_crop_y:real_crop_y+crop_size,real_crop_x:real_crop_x+crop_size]
            batch_fake_ = data_fake_pixels[fake_batch_ids,:,fake_crop_y:fake_crop_y+crop_size,fake_crop_x:fake_crop_x+crop_size]

            batch_real = torch.from_numpy(batch_real_).to(device)/255.0
            batch_fake = torch.from_numpy(batch_fake_).to(device)/255.0

            logits_real = resnet(batch_real)
            logits_fake = resnet(batch_fake)

            real_score = torch.mean((logits_real>0).float())
            fake_score = torch.mean((logits_fake<0).float())

            avg_real_score += real_score.item()
            avg_fake_score += fake_score.item()
            avg_num += 1

    avg_real_score = avg_real_score/avg_num
    avg_fake_score = avg_fake_score/avg_num
    result_Cls_score = (avg_real_score + avg_fake_score)/2

    #write result_Cls_score
    fout = open(output_dir+"/result_Cls_score.txt", 'w')
    fout.write("real_score:\n"+str(avg_real_score)+"\n")
    fout.write("fake_score:\n"+str(avg_fake_score)+"\n")
    fout.write("Cls_score:\n"+str(result_Cls_score)+"\n")
    fout.close()


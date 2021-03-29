import os
import time
import math
import random
import numpy as np
import h5py
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.linalg import sqrtm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

import cutils
import binvox_rw_faster as binvox_rw

from utils import *


max_num_of_styles = 16
max_num_of_contents = 100

shapenet_dir = "/local-scratch2/ShapeNetCore.v1.lite/"
#shapenet_dir = "/home/zhiqinc/zhiqinc/ShapeNetCore.v1.lite/"


class ShapeNet(torch.utils.data.Dataset):
    def __init__(self, imgdir, voxel_size):
        self.imgdir = imgdir
        self.voxel_size = voxel_size

        class_names = os.listdir(self.imgdir)
        class_names = [name for name in class_names if name[0]=='0' and len(name)==8]
        self.img_names = []
        self.img_class = []
        for i in range(len(class_names)):
            classname = class_names[i]
            tmp = os.listdir(self.imgdir+classname)
            tmp = [self.imgdir+classname+"/"+name+"/model_depth_fusion.binvox" for name in tmp]
            self.img_names += tmp
            self.img_class += [i] * len(tmp)
        print("dataset_len:", len(self.img_names))
        print("class_num:", len(class_names))
        print("voxel_size:", self.voxel_size)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        if self.voxel_size==128:
            tmp_raw = get_vox_from_binvox_1over2_return_small(self.img_names[index]).astype(np.float32)
            tmp_raw = np.reshape(tmp_raw, [1,128,128,128])
        elif self.voxel_size==256:
            tmp_raw = get_vox_from_binvox(self.img_names[index]).astype(np.float32)
            tmp_raw = np.reshape(tmp_raw, [1,256,256,256])
        class_id = np.array([self.img_class[index]], np.int64)
        return tmp_raw, class_id

class classifier(nn.Module):
    def __init__(self, ef_dim, z_dim, class_num, voxel_size):
        super(classifier, self).__init__()
        self.ef_dim = ef_dim
        self.z_dim = z_dim
        self.class_num = class_num
        self.voxel_size = voxel_size

        self.conv_1 = nn.Conv3d(1,             self.ef_dim,   4, stride=2, padding=1, bias=True)
        self.bn_1 = nn.InstanceNorm3d(self.ef_dim)

        self.conv_2 = nn.Conv3d(self.ef_dim,   self.ef_dim*2, 4, stride=2, padding=1, bias=True)
        self.bn_2 = nn.InstanceNorm3d(self.ef_dim*2)

        self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=True)
        self.bn_3 = nn.InstanceNorm3d(self.ef_dim*4)

        self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=True)
        self.bn_4 = nn.InstanceNorm3d(self.ef_dim*8)

        self.conv_5 = nn.Conv3d(self.ef_dim*8, self.z_dim, 4, stride=2, padding=1, bias=True)

        if self.voxel_size==256:
            self.bn_5 = nn.InstanceNorm3d(self.z_dim)
            self.conv_5_2 = nn.Conv3d(self.z_dim, self.z_dim, 4, stride=2, padding=1, bias=True)

        self.linear1 = nn.Linear(self.z_dim, self.class_num, bias=True)



    def forward(self, inputs, is_training=False):
        out = inputs

        out = self.bn_1(self.conv_1(out))
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.bn_2(self.conv_2(out))
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.bn_3(self.conv_3(out))
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.bn_4(self.conv_4(out))
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_5(out)

        if self.voxel_size==256:
            out = self.bn_5(out)
            out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
            out = self.conv_5_2(out)

        z = F.adaptive_avg_pool3d(out, output_size=(1, 1, 1))
        z = z.view(-1,self.z_dim)
        out = F.leaky_relu(z, negative_slope=0.01, inplace=True)
        
        out = self.linear1(out)

        return out, z


def train_classifier(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')


    class_names = os.listdir(shapenet_dir)
    class_names = [name for name in class_names if name[0]=='0' and len(name)==8]
    class_num = len(class_names)

    if config.output_size==128:
        voxel_size = 128
        batch_size = 32
    elif config.output_size==256:
        voxel_size = 256
        batch_size = 4

    z_dim = 512
    num_epochs = 100

    dataloader = torch.utils.data.DataLoader(ShapeNet(shapenet_dir, voxel_size), batch_size=batch_size, shuffle=True, num_workers=16)

    Clsshapenet = classifier(32, z_dim, class_num, voxel_size)
    Clsshapenet.to(device)
    optimizer = torch.optim.Adam(Clsshapenet.parameters(), lr=0.0001, betas=(0.5, 0.999))

    print("start")

    start_time = time.time()
    for epoch in range(num_epochs):

        Clsshapenet.train()

        avg_acc = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            voxels_, labels_ = data
            voxels = voxels_.to(device)
            labels = labels_.to(device).squeeze()

            optimizer.zero_grad()
            pred_labels, _ = Clsshapenet(voxels)
            loss = F.cross_entropy(pred_labels, labels)
            loss.backward()
            optimizer.step()

            acc = torch.mean((torch.max(pred_labels, 1)[1]==labels).float())
            
            avg_acc += acc.item()
            avg_count += 1

        print('Epoch: [%d/%d] time: %.0f, accuracy: %.8f' % (epoch, num_epochs, time.time()-start_time, avg_acc/avg_count))
        torch.save(Clsshapenet.state_dict(), 'Clsshapenet_'+str(voxel_size)+'.pth')




def compute_FID_for_real(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    class_num = 24

    data_dir = config.data_dir

    #load style shapes
    fin = open("splits/"+config.data_style+".txt")
    styleset_names = [name.strip() for name in fin.readlines()]
    fin.close()
    styleset_len = len(styleset_names)

    #load content shapes
    fin = open("splits/"+config.data_content+".txt")
    dataset_names = [name.strip() for name in fin.readlines()]
    fin.close()
    dataset_len = len(dataset_names)

    if config.output_size==128:
        voxel_size = 128
    elif config.output_size==256:
        voxel_size = 256

    z_dim = 512

    Clsshapenet = classifier(32, z_dim, class_num, voxel_size)
    Clsshapenet.to(device)
    Clsshapenet.load_state_dict(torch.load('Clsshapenet_'+str(voxel_size)+'.pth'))

    activation_all = np.zeros([dataset_len, z_dim])
    activation_style = np.zeros([max_num_of_styles, z_dim])


    for content_id in range(dataset_len):
        print("processing content - "+str(content_id+1)+"/"+str(dataset_len))
        if config.output_size==128:
            tmp_raw = get_vox_from_binvox_1over2_return_small(os.path.join(data_dir,dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
        elif config.output_size==256:
            tmp_raw = get_vox_from_binvox(os.path.join(data_dir,dataset_names[content_id]+"/model_depth_fusion.binvox")).astype(np.uint8)

        voxels = torch.from_numpy(tmp_raw).to(device).unsqueeze(0).unsqueeze(0).float()
        _, z = Clsshapenet(voxels)
        activation_all[content_id] = z.view(-1).detach().cpu().numpy()


    for style_id in range(max_num_of_styles):
        print("processing style - "+str(style_id+1)+"/"+str(styleset_len))
        if config.output_size==128:
            tmp_raw = get_vox_from_binvox_1over2_return_small(os.path.join(data_dir,styleset_names[style_id]+"/model_depth_fusion.binvox")).astype(np.uint8)
        elif config.output_size==256:
            tmp_raw = get_vox_from_binvox(os.path.join(data_dir,styleset_names[style_id]+"/model_depth_fusion.binvox")).astype(np.uint8)

        voxels = torch.from_numpy(tmp_raw).to(device).unsqueeze(0).unsqueeze(0).float()
        _, z = Clsshapenet(voxels)
        activation_style[style_id] = z.view(-1).detach().cpu().numpy()

    mu_all, sigma_all = np.mean(activation_all, axis=0), np.cov(activation_all, rowvar=False)
    mu_style, sigma_style = np.mean(activation_style, axis=0), np.cov(activation_style, rowvar=False)

    hdf5_file = h5py.File("precomputed_real_mu_sigma_"+str(voxel_size)+"_"+config.data_content+"_num_style_"+str(max_num_of_styles)+".hdf5", mode='w')
    hdf5_file.create_dataset("mu_all", mu_all.shape, np.float32)
    hdf5_file.create_dataset("sigma_all", sigma_all.shape, np.float32)
    hdf5_file.create_dataset("mu_style", mu_style.shape, np.float32)
    hdf5_file.create_dataset("sigma_style", sigma_style.shape, np.float32)

    hdf5_file["mu_all"][:] = mu_all
    hdf5_file["sigma_all"][:] = sigma_all
    hdf5_file["mu_style"][:] = mu_style
    hdf5_file["sigma_style"][:] = sigma_style
    
    hdf5_file.close()





def eval_FID(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    class_num = 24

    result_dir = "output_for_FID"
    if not os.path.exists(result_dir):
        print("ERROR: result_dir does not exist! "+result_dir)
        exit(-1)

    output_dir = "eval_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if config.output_size==128:
        voxel_size = 128
    elif config.output_size==256:
        voxel_size = 256

    z_dim = 512

    Clsshapenet = classifier(32, z_dim, class_num, voxel_size)
    Clsshapenet.to(device)
    Clsshapenet.load_state_dict(torch.load('Clsshapenet_'+str(voxel_size)+'.pth'))

    num_of_shapes = max_num_of_styles*max_num_of_contents
    activation_test = np.zeros([num_of_shapes, z_dim])

    counter = 0
    for content_id in range(max_num_of_contents):
        print("processing content - "+str(content_id+1)+"/"+str(max_num_of_contents))
        for style_id in range(max_num_of_styles):
            voxel_model_file = open(result_dir+"/output_content_"+str(content_id)+"_style_"+str(style_id)+".binvox", 'rb')
            tmp_raw = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=False).data.astype(np.uint8)
            if config.output_size==128:
                tmp_raw = tmp_raw[64:-64,64:-64,64:-64]

            voxels = torch.from_numpy(tmp_raw).to(device).unsqueeze(0).unsqueeze(0).float()
            _, z = Clsshapenet(voxels)
            activation_test[counter] = z.view(-1).detach().cpu().numpy()
            counter += 1

    mu_test, sigma_test = np.mean(activation_test, axis=0), np.cov(activation_test, rowvar=False)

    hdf5_file = h5py.File("precomputed_real_mu_sigma_"+str(voxel_size)+"_"+config.data_content+"_num_style_"+str(max_num_of_styles)+".hdf5", mode='r')
    mu_all = hdf5_file["mu_all"][:]
    sigma_all = hdf5_file["sigma_all"][:]
    mu_style = hdf5_file["mu_style"][:]
    sigma_style = hdf5_file["sigma_style"][:]
    hdf5_file.close()

    def calculate_fid(mu1, sigma1, mu2, sigma2):
        ssdiff = np.sum((mu1 - mu2)**2)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    
    fid_all = calculate_fid(mu_test, sigma_test, mu_all, sigma_all)
    fid_style = calculate_fid(mu_test, sigma_test, mu_style, sigma_style)

    #write result_Cls_score
    fout = open(output_dir+"/result_FID.txt", 'w')
    fout.write("fid_all:\n"+str(fid_all)+"\n")
    fout.write("fid_style:\n"+str(fid_style)+"\n")
    fout.close()


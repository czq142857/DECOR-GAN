import numpy as np
import cv2
import os
from scipy.io import loadmat
import random
import time
import math
import binvox_rw


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable



def get_vox_from_binvox(objname):

    #get voxel models
    voxel_model_file = open(objname, 'rb')
    voxel_model_512 = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=True).data.astype(np.uint8)
    step_size = 2
    voxel_model_256 = voxel_model_512[0::step_size,0::step_size,0::step_size]
    for i in range(step_size):
        for j in range(step_size):
            for k in range(step_size):
                voxel_model_256 = np.maximum(voxel_model_256,voxel_model_512[i::step_size,j::step_size,k::step_size])
    #add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
    #voxel_model_256 = np.flip(np.transpose(voxel_model_256, (2,1,0)),2)

    return voxel_model_256


def get_vox_from_binvox_1over2(objname):

    #get voxel models
    voxel_model_file = open(objname, 'rb')
    voxel_model_512 = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=True).data.astype(np.uint8)
    step_size = 4
    padding_size = 256%step_size
    output_padding = 128-(256//step_size)
    #voxel_model_512 = voxel_model_512[padding_size:-padding_size,padding_size:-padding_size,padding_size:-padding_size]
    voxel_model_128 = voxel_model_512[0::step_size,0::step_size,0::step_size]
    for i in range(step_size):
        for j in range(step_size):
            for k in range(step_size):
                voxel_model_128 = np.maximum(voxel_model_128,voxel_model_512[i::step_size,j::step_size,k::step_size])
    #add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
    #voxel_model_128 = np.flip(np.transpose(voxel_model_128, (2,1,0)),2)
    voxel_model_256 = np.zeros([256,256,256],np.uint8)
    voxel_model_256[output_padding:-output_padding,output_padding:-output_padding,output_padding:-output_padding] = voxel_model_128

    return voxel_model_256


def get_vox_from_binvox_1over2_return_small(objname):

    #get voxel models
    voxel_model_file = open(objname, 'rb')
    voxel_model_512 = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=True).data.astype(np.uint8)
    step_size = 4
    padding_size = 256%step_size
    output_padding = 128-(256//step_size)
    #voxel_model_512 = voxel_model_512[padding_size:-padding_size,padding_size:-padding_size,padding_size:-padding_size]
    voxel_model_128 = voxel_model_512[0::step_size,0::step_size,0::step_size]
    for i in range(step_size):
        for j in range(step_size):
            for k in range(step_size):
                voxel_model_128 = np.maximum(voxel_model_128,voxel_model_512[i::step_size,j::step_size,k::step_size])
    #add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
    #voxel_model_128 = np.flip(np.transpose(voxel_model_128, (2,1,0)),2)

    return voxel_model_128



def write_ply_point(name, vertices):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    fout.close()


def write_ply_point_normal(name, vertices, normals=None):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property float nx\n")
    fout.write("property float ny\n")
    fout.write("property float nz\n")
    fout.write("end_header\n")
    if normals is None:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(vertices[ii,3])+" "+str(vertices[ii,4])+" "+str(vertices[ii,5])+"\n")
    else:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(normals[ii,0])+" "+str(normals[ii,1])+" "+str(normals[ii,2])+"\n")
    fout.close()


def write_ply_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face "+str(len(triangles))+"\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
    fout.close()


def write_obj_triangle(name, vertices, triangles):
	fout = open(name, 'w')
	for ii in range(len(vertices)):
		fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	for ii in range(len(triangles)):
		fout.write("f "+str(int(triangles[ii,0]+1))+" "+str(int(triangles[ii,1]+1))+" "+str(int(triangles[ii,2]+1))+"\n")
	fout.close()


class voxel_renderer:
    def __init__(self, render_IO_vox_size=256, render_boundary_padding_size=16):
        self.render_IO_vox_size = render_IO_vox_size
        self.render_boundary_padding_size = render_boundary_padding_size
        self.render_fix_vox_size = self.render_IO_vox_size + self.render_boundary_padding_size*2
        self.voxel_idxs = np.linspace(-self.render_fix_vox_size/2+0.5, self.render_fix_vox_size/2-0.5, self.render_fix_vox_size, dtype = np.float32)
        self.voxel_x, self.voxel_y, self.voxel_z = np.meshgrid(self.voxel_idxs,self.voxel_idxs,self.voxel_idxs, sparse=False, indexing='ij')

    def render_img(self, voxel_in, threshold, view=0, get_depth=False, ray_x = 0, ray_y = 0, ray_z = 1, steep_threshold = 16):
        imgsize = voxel_in.shape[0]
        if self.render_IO_vox_size!=imgsize:
            print("ERROR: render_img() voxel size does not match!", imgsize, self.render_IO_vox_size)
            exit(-1)

        voxel = np.zeros([self.render_fix_vox_size,self.render_fix_vox_size,self.render_fix_vox_size], np.uint8)
        voxel[self.render_boundary_padding_size:-self.render_boundary_padding_size,self.render_boundary_padding_size:-self.render_boundary_padding_size,self.render_boundary_padding_size:-self.render_boundary_padding_size] = (voxel_in>threshold).astype(np.uint8)[::-1,::-1]

        #get mask and depth

        if view==0: #x-y-z

            new_x2 = (  self.voxel_x - self.voxel_y)/2**0.5
            new_y2 = (  self.voxel_x + self.voxel_y)/2**0.5
            new_z2 = self.voxel_z

            new_x = (  new_x2 + new_z2)/2**0.5 + (self.render_fix_vox_size/2 + 0.01)
            new_y = new_y2 + (self.render_fix_vox_size/2 + 0.01)
            new_z = (  new_x2 - new_z2)/2**0.5 + (self.render_fix_vox_size/2 + 0.01)

            new_x = np.clip(new_x.astype(np.int32), 0, self.render_fix_vox_size-1)
            new_y = np.clip(new_y.astype(np.int32), 0, self.render_fix_vox_size-1)
            new_z = np.clip(new_z.astype(np.int32), 0, self.render_fix_vox_size-1)
            voxel = voxel[new_x,new_y,new_z]
            mask = np.amax(voxel, axis=0).astype(np.int32)
            depth = np.argmax(voxel,axis=0)
        if view==1: #right-top
            new_x = (- self.voxel_x + self.voxel_y)/2**0.5 + (self.render_fix_vox_size/2 + 0.01)
            new_y = (  self.voxel_x + self.voxel_y)/2**0.5 + (self.render_fix_vox_size/2 + 0.01)
            new_z = self.voxel_z + (self.render_fix_vox_size/2 + 0.01)
            new_x = np.clip(new_x.astype(np.int32), 0, self.render_fix_vox_size-1)
            new_y = np.clip(new_y.astype(np.int32), 0, self.render_fix_vox_size-1)
            new_z = np.clip(new_z.astype(np.int32), 0, self.render_fix_vox_size-1)
            voxel = voxel[new_x,new_y,new_z]
            mask = np.amax(voxel, axis=0).astype(np.int32)
            depth = np.argmax(voxel,axis=0)
        if view==2: #front-top
            new_x = self.voxel_x + (self.render_fix_vox_size/2 + 0.01)
            new_y = (- self.voxel_y + self.voxel_z)/2**0.5 + (self.render_fix_vox_size/2 + 0.01)
            new_z = (  self.voxel_y + self.voxel_z)/2**0.5 + (self.render_fix_vox_size/2 + 0.01)
            new_x = np.clip(new_x.astype(np.int32), 0, self.render_fix_vox_size-1)
            new_y = np.clip(new_y.astype(np.int32), 0, self.render_fix_vox_size-1)
            new_z = np.clip(new_z.astype(np.int32), 0, self.render_fix_vox_size-1)
            voxel = voxel[new_x,new_y,new_z]
            mask = np.amax(voxel, axis=2).astype(np.int32)
            depth = np.argmax(voxel,axis=2)
            mask = np.flip(np.transpose(mask, (1,0)),0)
            depth = np.flip(np.transpose(depth, (1,0)),0)
        if view==3: #left-front
            new_x = (  self.voxel_x + self.voxel_z)/2**0.5 + (self.render_fix_vox_size/2 + 0.01)
            new_y = self.voxel_y + (self.render_fix_vox_size/2 + 0.01)
            new_z = (  self.voxel_x - self.voxel_z)/2**0.5 + (self.render_fix_vox_size/2 + 0.01)
            new_x = np.clip(new_x.astype(np.int32), 0, self.render_fix_vox_size-1)
            new_y = np.clip(new_y.astype(np.int32), 0, self.render_fix_vox_size-1)
            new_z = np.clip(new_z.astype(np.int32), 0, self.render_fix_vox_size-1)
            voxel = voxel[new_x,new_y,new_z]
            mask = np.amax(voxel, axis=0).astype(np.int32)
            depth = np.argmax(voxel,axis=0)
        if view==4: #left
            mask = np.amax(voxel, axis=0).astype(np.int32)
            depth = np.argmax(voxel,axis=0)
        if view==5: #top
            mask = np.amax(voxel, axis=1).astype(np.int32)
            depth = np.argmax(voxel,axis=1)
            mask = np.transpose(mask, (1,0))
            depth = np.transpose(depth, (1,0))
        if view==6: #front
            mask = np.amax(voxel, axis=2).astype(np.int32)
            depth = np.argmax(voxel,axis=2)
            mask = np.transpose(mask, (1,0))
            depth = np.transpose(depth, (1,0))

        depth = depth+(1-mask)*512

        #visualize

        if get_depth: #depth
            output = 255 + np.min(depth) - depth
            output = np.clip(output, 0,255).astype(np.uint8)
        else: #surface
            output = np.ones([self.render_fix_vox_size,self.render_fix_vox_size],np.float32)
            dx = depth[:,:-1] - depth[:,1:]
            dy = depth[:-1,:] - depth[1:,:]
            dxp = 0
            dyp = 0
            counter = 0
            #1. get normal
            # 1 - 2 - 3
            # | \ | / |
            # 4 - 5 - 6
            # | / | \ |
            # 7 - 8 - 9
            for iii in range(12):
                if iii==0: #/\ 12 25
                    partial_dx = dx[:-2,:-1]
                    partial_dy = dy[:-1,1:-1]
                elif iii==1: #/\ 45 14
                    partial_dx = dx[1:-1,:-1]
                    partial_dy = dy[:-1,:-2]
                elif iii==2: #/\ 45 25
                    partial_dx = dx[1:-1,:-1]
                    partial_dy = dy[:-1,1:-1]
                elif iii==3: #/\ 23 25
                    partial_dx = dx[:-2,1:]
                    partial_dy = dy[:-1,1:-1]
                elif iii==4: #/\ 56 25
                    partial_dx = dx[1:-1,1:]
                    partial_dy = dy[:-1,1:-1]
                elif iii==5: #/\ 56 36
                    partial_dx = dx[1:-1,1:]
                    partial_dy = dy[:-1,2:]
                elif iii==6: #/\ 56 69
                    partial_dx = dx[1:-1,1:]
                    partial_dy = dy[1:,2:]
                elif iii==7: #/\ 56 58
                    partial_dx = dx[1:-1,1:]
                    partial_dy = dy[1:,1:-1]
                elif iii==8: #/\ 89 58
                    partial_dx = dx[2:,1:]
                    partial_dy = dy[1:,1:-1]
                elif iii==9: #/\ 78 58
                    partial_dx = dx[2:,:-1]
                    partial_dy = dy[1:,1:-1]
                elif iii==10: #/\ 45 58
                    partial_dx = dx[1:-1,:-1]
                    partial_dy = dy[1:,1:-1]
                elif iii==11: #/\ 45 47
                    partial_dx = dx[1:-1,:-1]
                    partial_dy = dy[1:,:-2]
                partial_m = (np.abs(partial_dx)<steep_threshold) & (np.abs(partial_dy)<steep_threshold)
                dxp = dxp+partial_dx*partial_m
                dyp = dyp+partial_dy*partial_m
                counter = counter+partial_m
            
            counter = np.maximum(counter,1)
            dxp = dxp/counter
            dyp = dyp/counter

            ds = np.sqrt(dxp**2 + dyp**2 + 1)
            dxp = dxp/ds
            dyp = dyp/ds
            dzp = 1.0/ds

            output[1:-1,1:-1] = dxp*ray_x + dyp*ray_y + dzp*ray_z

            output = output*220 + (1-mask)*256
            output = output[self.render_boundary_padding_size:-self.render_boundary_padding_size,self.render_boundary_padding_size:-self.render_boundary_padding_size]
            output = np.clip(output, 0,255).astype(np.uint8)

        return output


    def render_img_with_camera_pose(self, voxel_in, threshold, cam_alpha = 0.785, cam_beta = 0.785, get_depth = False, processed = False, ray_x = 0, ray_y = 0, ray_z = 1, steep_threshold = 16):
        imgsize = voxel_in.shape[0]

        if processed:
            if self.render_fix_vox_size!=imgsize:
                print("ERROR: render_img() voxel size does not match!", imgsize, self.render_fix_vox_size)
                exit(-1)
            voxel = (voxel_in>threshold).astype(np.uint8)
        else:
            if self.render_IO_vox_size!=imgsize:
                print("ERROR: render_img() voxel size does not match!", imgsize, self.render_IO_vox_size)
                exit(-1)
            voxel = np.zeros([self.render_fix_vox_size,self.render_fix_vox_size,self.render_fix_vox_size], np.uint8)
            voxel[self.render_boundary_padding_size:-self.render_boundary_padding_size,self.render_boundary_padding_size:-self.render_boundary_padding_size,self.render_boundary_padding_size:-self.render_boundary_padding_size] = (voxel_in>threshold).astype(np.uint8)[::-1,::-1]

        #get mask and depth
        sin_alpha = np.sin(cam_alpha)
        cos_alpha = np.cos(cam_alpha)
        sin_beta = np.sin(cam_beta)
        cos_beta = np.cos(cam_beta)

        new_x2 = cos_beta*self.voxel_x - sin_beta*self.voxel_y
        new_y2 = sin_beta*self.voxel_x + cos_beta*self.voxel_y
        new_z2 = self.voxel_z

        new_x = sin_alpha*new_x2 + cos_alpha*new_z2 + (self.render_fix_vox_size/2 + 0.01)
        new_y = new_y2 + (self.render_fix_vox_size/2 + 0.01)
        new_z = cos_alpha*new_x2 - sin_alpha*new_z2 + (self.render_fix_vox_size/2 + 0.01)

        new_x = np.clip(new_x.astype(np.int32), 0, self.render_fix_vox_size-1)
        new_y = np.clip(new_y.astype(np.int32), 0, self.render_fix_vox_size-1)
        new_z = np.clip(new_z.astype(np.int32), 0, self.render_fix_vox_size-1)
        voxel = voxel[new_x,new_y,new_z]
        mask = np.amax(voxel, axis=0).astype(np.int32)
        depth = np.argmax(voxel,axis=0)

        depth = depth+(1-mask)*512

        #visualize

        if get_depth: #depth
            output = 255 + np.min(depth) - depth
            output = np.clip(output, 0,255).astype(np.uint8)
        else: #surface
            dx = depth[:,:-1] - depth[:,1:]
            dy = depth[:-1,:] - depth[1:,:]
            dxp = 0
            dyp = 0
            counter = 0
            #1. get normal
            # 1 - 2 - 3
            # | \ | / |
            # 4 - 5 - 6
            # | / | \ |
            # 7 - 8 - 9
            for iii in range(12):
                if iii==0: #/\ 12 25
                    partial_dx = dx[:-2,:-1]
                    partial_dy = dy[:-1,1:-1]
                elif iii==1: #/\ 45 14
                    partial_dx = dx[1:-1,:-1]
                    partial_dy = dy[:-1,:-2]
                elif iii==2: #/\ 45 25
                    partial_dx = dx[1:-1,:-1]
                    partial_dy = dy[:-1,1:-1]
                elif iii==3: #/\ 23 25
                    partial_dx = dx[:-2,1:]
                    partial_dy = dy[:-1,1:-1]
                elif iii==4: #/\ 56 25
                    partial_dx = dx[1:-1,1:]
                    partial_dy = dy[:-1,1:-1]
                elif iii==5: #/\ 56 36
                    partial_dx = dx[1:-1,1:]
                    partial_dy = dy[:-1,2:]
                elif iii==6: #/\ 56 69
                    partial_dx = dx[1:-1,1:]
                    partial_dy = dy[1:,2:]
                elif iii==7: #/\ 56 58
                    partial_dx = dx[1:-1,1:]
                    partial_dy = dy[1:,1:-1]
                elif iii==8: #/\ 89 58
                    partial_dx = dx[2:,1:]
                    partial_dy = dy[1:,1:-1]
                elif iii==9: #/\ 78 58
                    partial_dx = dx[2:,:-1]
                    partial_dy = dy[1:,1:-1]
                elif iii==10: #/\ 45 58
                    partial_dx = dx[1:-1,:-1]
                    partial_dy = dy[1:,1:-1]
                elif iii==11: #/\ 45 47
                    partial_dx = dx[1:-1,:-1]
                    partial_dy = dy[1:,:-2]
                partial_m = (np.abs(partial_dx)<steep_threshold) & (np.abs(partial_dy)<steep_threshold)
                dxp = dxp+partial_dx*partial_m
                dyp = dyp+partial_dy*partial_m
                counter = counter+partial_m
            
            counter = np.maximum(counter,1)
            dxp = dxp/counter
            dyp = dyp/counter

            ds = np.sqrt(dxp**2 + dyp**2 + 1)
            dxp = dxp/ds
            dyp = dyp/ds
            dzp = 1.0/ds

            output = dxp*ray_x + dyp*ray_y + dzp*ray_z

            output = output*220 + (1-mask[1:-1,1:-1])*256
            output = output[self.render_boundary_padding_size-1:-self.render_boundary_padding_size+1,self.render_boundary_padding_size-1:-self.render_boundary_padding_size+1]
            output = np.clip(output, 0,255).astype(np.uint8)

        return output

    def use_gpu(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.voxel_x_tensor = torch.from_numpy(self.voxel_x/(self.render_fix_vox_size/2)).to(self.device)
        self.voxel_y_tensor = torch.from_numpy(self.voxel_y/(self.render_fix_vox_size/2)).to(self.device)
        self.voxel_z_tensor = torch.from_numpy(self.voxel_z/(self.render_fix_vox_size/2)).to(self.device)

    def render_img_with_camera_pose_gpu(self, voxel_in, threshold, cam_alpha = 0.785, cam_beta = 0.785, get_depth = False, processed = False, ray_x = 0, ray_y = 0, ray_z = 1, steep_threshold = 16):
        if processed:
            voxel_tensor = voxel_in
            imgsize = voxel_tensor.size()[2]
            if self.render_fix_vox_size!=imgsize:
                print("ERROR: render_img() voxel size does not match!", imgsize, self.render_fix_vox_size)
                exit(-1)
        else:
            imgsize = voxel_in.shape[0]
            if self.render_IO_vox_size!=imgsize:
                print("ERROR: render_img() voxel size does not match!", imgsize, self.render_IO_vox_size)
                exit(-1)
            voxel = np.zeros([self.render_fix_vox_size,self.render_fix_vox_size,self.render_fix_vox_size], np.float32)
            voxel[self.render_boundary_padding_size:-self.render_boundary_padding_size,self.render_boundary_padding_size:-self.render_boundary_padding_size,self.render_boundary_padding_size:-self.render_boundary_padding_size] = voxel_in[::-1,::-1]
            voxel_tensor = torch.from_numpy(voxel).to(self.device).unsqueeze(0).unsqueeze(0).float()

        #get mask and depth
        sin_alpha = np.sin(cam_alpha)
        cos_alpha = np.cos(cam_alpha)
        sin_beta = np.sin(cam_beta)
        cos_beta = np.cos(cam_beta)

        new_x2_tensor = cos_beta*self.voxel_x_tensor - sin_beta*self.voxel_y_tensor
        new_y2_tensor = sin_beta*self.voxel_x_tensor + cos_beta*self.voxel_y_tensor
        new_z2_tensor = self.voxel_z_tensor

        new_x_tensor = sin_alpha*new_x2_tensor + cos_alpha*new_z2_tensor
        new_y_tensor = new_y2_tensor
        new_z_tensor = cos_alpha*new_x2_tensor - sin_alpha*new_z2_tensor

        new_xyz_tensor = torch.cat([new_x_tensor.unsqueeze(3),new_y_tensor.unsqueeze(3),new_z_tensor.unsqueeze(3)], 3).unsqueeze(0)

        voxel_tensor = F.grid_sample(voxel_tensor, new_xyz_tensor, mode='bilinear', padding_mode='zeros').squeeze()
        voxel_tensor = voxel_tensor>threshold
        mask, depth = torch.max(voxel_tensor,0)

        mask = mask.float()
        depth = depth.float()
        depth = depth+(1-mask)*512


        #visualize

        if get_depth: #depth
            output = 255 + torch.min(depth) - depth
            output = torch.clamp(output, min=0, max=255)
            output = output.detach().cpu().numpy().astype(np.uint8)
        else: #surface
            dx = depth[:,:-1] - depth[:,1:]
            dy = depth[:-1,:] - depth[1:,:]
            dxp = 0
            dyp = 0
            counter = 0
            #1. get normal
            # 1 - 2 - 3
            # | \ | / |
            # 4 - 5 - 6
            # | / | \ |
            # 7 - 8 - 9
            for iii in range(12):
                if iii==0: #/\ 12 25
                    partial_dx = dx[:-2,:-1]
                    partial_dy = dy[:-1,1:-1]
                elif iii==1: #/\ 45 14
                    partial_dx = dx[1:-1,:-1]
                    partial_dy = dy[:-1,:-2]
                elif iii==2: #/\ 45 25
                    partial_dx = dx[1:-1,:-1]
                    partial_dy = dy[:-1,1:-1]
                elif iii==3: #/\ 23 25
                    partial_dx = dx[:-2,1:]
                    partial_dy = dy[:-1,1:-1]
                elif iii==4: #/\ 56 25
                    partial_dx = dx[1:-1,1:]
                    partial_dy = dy[:-1,1:-1]
                elif iii==5: #/\ 56 36
                    partial_dx = dx[1:-1,1:]
                    partial_dy = dy[:-1,2:]
                elif iii==6: #/\ 56 69
                    partial_dx = dx[1:-1,1:]
                    partial_dy = dy[1:,2:]
                elif iii==7: #/\ 56 58
                    partial_dx = dx[1:-1,1:]
                    partial_dy = dy[1:,1:-1]
                elif iii==8: #/\ 89 58
                    partial_dx = dx[2:,1:]
                    partial_dy = dy[1:,1:-1]
                elif iii==9: #/\ 78 58
                    partial_dx = dx[2:,:-1]
                    partial_dy = dy[1:,1:-1]
                elif iii==10: #/\ 45 58
                    partial_dx = dx[1:-1,:-1]
                    partial_dy = dy[1:,1:-1]
                elif iii==11: #/\ 45 47
                    partial_dx = dx[1:-1,:-1]
                    partial_dy = dy[1:,:-2]
                partial_m = (torch.abs(partial_dx)<steep_threshold) & (torch.abs(partial_dy)<steep_threshold)
                partial_m = partial_m.float()
                dxp = dxp+partial_dx*partial_m
                dyp = dyp+partial_dy*partial_m
                counter = counter+partial_m
            
            counter = torch.clamp(counter,min=1)
            dxp = dxp/counter
            dyp = dyp/counter

            ds = torch.sqrt(dxp**2 + dyp**2 + 1)
            dxp = dxp/ds
            dyp = dyp/ds
            dzp = 1.0/ds

            output = dxp*ray_x + dyp*ray_y + dzp*ray_z

            output = output*220 + (1-mask[1:-1,1:-1])*256
            output = output[self.render_boundary_padding_size-1:-self.render_boundary_padding_size+1,self.render_boundary_padding_size-1:-self.render_boundary_padding_size+1]
            output = torch.clamp(output, min=0, max=255)
            output = output.detach().cpu().numpy().astype(np.uint8)

        return output

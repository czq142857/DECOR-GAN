import os
import time
import math
import random
import numpy as np
import h5py
import cv2

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
max_num_of_contents = 20


def eval_IOU(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    result_dir = "output_for_eval"
    if not os.path.exists(result_dir):
        print("ERROR: result_dir does not exist! "+result_dir)
        exit(-1)

    output_dir = "eval_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #load style shapes
    fin = open("splits/"+config.data_style+".txt")
    styleset_names = [name.strip() for name in fin.readlines()]
    fin.close()
    styleset_len_original = len(styleset_names)
    styleset_len = min(styleset_len_original, max_num_of_styles)

    #load content shapes
    fin = open("splits/"+config.data_content+".txt")
    dataset_names = [name.strip() for name in fin.readlines()]
    fin.close()
    dataset_len_original = len(dataset_names)
    dataset_len = min(dataset_len_original, max_num_of_contents)



    result_IOU_strict = np.zeros([dataset_len,styleset_len],np.float32)
    result_IOU_loose = np.zeros([dataset_len,styleset_len],np.float32)

    for content_id in range(dataset_len):
        voxel_model_file = open(result_dir+"/content_"+str(content_id)+"_coarse.binvox", 'rb')
        content_shape_coarse = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=False).data.astype(np.uint8)
        if not config.asymmetry:
            content_shape_coarse = content_shape_coarse[:,:,2:] #keep only symmetric part

        for style_id in range(styleset_len):
            voxel_model_file = open(result_dir+"/output_content_"+str(content_id)+"_style_"+str(style_id)+".binvox", 'rb')
            output_shape = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=False).data.astype(np.uint8)

            output_shape_tensor = torch.from_numpy(output_shape).to(device).unsqueeze(0).unsqueeze(0).float()
            output_shape_coarse_tensor = F.max_pool3d(output_shape_tensor, kernel_size = 4, stride = 4, padding = 0)
            output_shape_coarse = output_shape_coarse_tensor.detach().cpu().numpy()[0,0]
            output_shape_coarse = np.round(output_shape_coarse).astype(np.uint8)
            if not config.asymmetry:
                output_shape_coarse = output_shape_coarse[:,:,2:] #keep only symmetric part

            IOU_strict = np.sum( output_shape_coarse & content_shape_coarse ) / float(np.sum( output_shape_coarse | content_shape_coarse ))
            IOU_loose = np.sum( output_shape_coarse & content_shape_coarse ) / float(np.sum( content_shape_coarse ))

            result_IOU_strict[content_id,style_id] = IOU_strict
            result_IOU_loose[content_id,style_id] = IOU_loose
    


    #write result_IOU_strict
    fout = open(output_dir+"/result_IOU_strict.txt", 'w')
    for content_id in range(dataset_len):
        for style_id in range(styleset_len):
            fout.write(str(result_IOU_strict[content_id,style_id]))
            if style_id!=styleset_len-1:
                fout.write("\t")
        if content_id!=dataset_len-1:
            fout.write("\n")
    fout.close()
    
    #write result_IOU_loose
    fout = open(output_dir+"/result_IOU_loose.txt", 'w')
    for content_id in range(dataset_len):
        for style_id in range(styleset_len):
            fout.write(str(result_IOU_loose[content_id,style_id]))
            if style_id!=styleset_len-1:
                fout.write("\t")
        if content_id!=dataset_len-1:
            fout.write("\n")
    fout.close()
    
    #write result_IOU_mean
    fout = open(output_dir+"/result_IOU_mean.txt", 'w')
    fout.write("IOU_strict:\n"+str(np.mean(result_IOU_strict))+"\n")
    fout.write("IOU_loose:\n"+str(np.mean(result_IOU_loose))+"\n")
    fout.close()



def precompute_unique_patches_per_style(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    result_dir = "output_for_eval"
    if not os.path.exists(result_dir):
        print("ERROR: result_dir does not exist! "+result_dir)
        exit(-1)
    
    patches_dir = "unique_patches"
    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)


    #load style shapes
    fin = open("splits/"+config.data_style+".txt")
    styleset_names = [name.strip() for name in fin.readlines()]
    fin.close()
    styleset_len = len(styleset_names)
    print("precompute_unique_patches:",config.data_style,styleset_len)


    buffer_size = 256*256*16 #change the buffer size if the input voxel is large
    patch_size = 12
    if not config.asymmetry:
        padding_size = 8 - patch_size//2
    else:
        padding_size = 0


    #prepare dictionary for style shapes
    for style_id in range(styleset_len):
        print(style_id, styleset_len)

        voxel_model_file = open(result_dir+"/style_"+str(style_id)+".binvox", 'rb')
        style_shape = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=False).data.astype(np.uint8)
        style_shape = np.ascontiguousarray(style_shape[:,:,padding_size:])

        style_shape_tensor = torch.from_numpy(style_shape).to(device).unsqueeze(0).unsqueeze(0).float()
        style_shape_edge_tensor = F.max_pool3d(-style_shape_tensor, kernel_size = 3, stride = 1, padding = 1) + style_shape_tensor
        style_shape_dilated_tensor = F.max_pool3d(style_shape_edge_tensor, kernel_size = 3, stride = 1, padding = 1)
        style_shape_edge = style_shape_edge_tensor.detach().cpu().numpy()[0,0]
        style_shape_edge = np.round(style_shape_edge).astype(np.uint8)
        style_shape_dilated = style_shape_dilated_tensor.detach().cpu().numpy()[0,0]
        style_shape_dilated = np.round(style_shape_dilated).astype(np.uint8)

        patches = np.zeros([buffer_size,patch_size,patch_size,patch_size], np.uint8)
        patches_edge = np.zeros([buffer_size,patch_size,patch_size,patch_size], np.uint8)
        patches_dilated = np.zeros([buffer_size,patch_size,patch_size,patch_size], np.uint8)
        patch_num = cutils.get_patches_edge_dilated(style_shape,style_shape_edge,style_shape_dilated,patches,patches_edge,patches_dilated,patch_size)

        patches = np.copy(patches[:patch_num])
        patches_edge = np.copy(patches_edge[:patch_num])
        patches_dilated = np.copy(patches_dilated[:patch_num])
        patches_tensor = torch.from_numpy(patches).to(device).view(patch_num,-1).bool()

        pointer = 0

        for patch_id in range(patch_num-1):
            notduplicated = torch.bitwise_xor(patches_tensor[patch_id:patch_id+1], patches_tensor[patch_id+1:]).any(1).all().item()
            if notduplicated:
                patches[pointer] = patches[patch_id]
                patches_edge[pointer] = patches_edge[patch_id]
                patches_dilated[pointer] = patches_dilated[patch_id]
                pointer += 1
        patch_id = patch_num-1
        patches[pointer] = patches[patch_id]
        patches_edge[pointer] = patches_edge[patch_id]
        patches_dilated[pointer] = patches_dilated[patch_id]
        pointer += 1

        print("before --", patch_num)
        print("after --", pointer)

        hdf5_file = h5py.File(patches_dir+"/style_"+str(style_id)+".hdf5", 'w')
        hdf5_file.create_dataset("patches", [pointer,patch_size,patch_size,patch_size], np.uint8, compression=9)
        hdf5_file.create_dataset("patches_edge", [pointer,patch_size,patch_size,patch_size], np.uint8, compression=9)
        hdf5_file.create_dataset("patches_dilated", [pointer,patch_size,patch_size,patch_size], np.uint8, compression=9)
        hdf5_file["patches"][:] = patches[:pointer]
        hdf5_file["patches_edge"][:] = patches_edge[:pointer]
        hdf5_file["patches_dilated"][:] = patches_dilated[:pointer]
        hdf5_file.close()



def precompute_unique_patches_all_styles(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    result_dir = "output_for_eval"
    if not os.path.exists(result_dir):
        print("ERROR: result_dir does not exist! "+result_dir)
        exit(-1)
    
    patches_dir = "unique_patches"
    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)


    #load style shapes
    fin = open("splits/"+config.data_style+".txt")
    styleset_names = [name.strip() for name in fin.readlines()]
    fin.close()
    styleset_len = len(styleset_names)
    print("precompute_unique_patches:",config.data_style,styleset_len)


    buffer_size = 256*256*16 #change the buffer size if the input voxel is large
    patch_size = 12
    if not config.asymmetry:
        padding_size = 8 - patch_size//2
    else:
        padding_size = 0


    #prepare dictionary for style shapes
    dict_style_patches = []
    dict_style_patches_edge = []
    dict_style_patches_dilated = []
    dict_style_patches_tensor = []
    patch_num_total = 0
    for style_id in range(styleset_len):
        data_dict = h5py.File(patches_dir+"/style_"+str(style_id)+".hdf5", 'r')
        patches = data_dict['patches'][:]
        patches_edge = data_dict['patches_edge'][:]
        patches_dilated = data_dict['patches_dilated'][:]
        data_dict.close()
        patch_num = len(patches)
        patch_num_total += patch_num

        dict_style_patches.append(patches)
        dict_style_patches_edge.append(patches_edge)
        dict_style_patches_dilated.append(patches_dilated)
        patches_tensor = torch.from_numpy(patches).to(device).view(patch_num,-1).bool()
        dict_style_patches_tensor.append(patches_tensor)

    pointer_total = 0
    dict_style_patches_len = []
    for style_id in range(styleset_len):
        print(style_id,styleset_len)
        this_patches = dict_style_patches[style_id]
        this_patches_edge = dict_style_patches_edge[style_id]
        this_patches_dilated = dict_style_patches_dilated[style_id]
        this_patches_tensor = dict_style_patches_tensor[style_id]
        patch_num = len(this_patches)

        pointer = 1

        for patch_id in range(1,patch_num):
            notduplicated_flag = True

            for compare_id in range(0,style_id+1):
                compare_patches_tensor = dict_style_patches_tensor[compare_id]

                if compare_id!=style_id:
                    notduplicated = torch.bitwise_xor(this_patches_tensor[patch_id:patch_id+1], compare_patches_tensor[:dict_style_patches_len[compare_id]]).any(1).all().item()
                else:
                    notduplicated = torch.bitwise_xor(this_patches_tensor[patch_id:patch_id+1], compare_patches_tensor[:pointer]).any(1).all().item()
                
                if not notduplicated:
                    notduplicated_flag = False
                    break

            if notduplicated_flag:
                this_patches[pointer] = this_patches[patch_id]
                this_patches_edge[pointer] = this_patches_edge[patch_id]
                this_patches_dilated[pointer] = this_patches_dilated[patch_id]
                this_patches_tensor.data[pointer] = this_patches_tensor.data[patch_id]
                pointer += 1

        pointer_total += pointer
        dict_style_patches_len.append(pointer)

        print("before --", patch_num, patch_num_total)
        print("after --", pointer, pointer_total)

    hdf5_file = h5py.File(patches_dir+"/global_unique_"+config.data_style+".hdf5", 'w')
    for style_id in range(styleset_len):
        pointer = dict_style_patches_len[style_id]
        hdf5_file.create_dataset("patches_"+str(style_id), [pointer,patch_size,patch_size,patch_size], np.uint8, compression=9)
        hdf5_file.create_dataset("patches_edge_"+str(style_id), [pointer,patch_size,patch_size,patch_size], np.uint8, compression=9)
        hdf5_file.create_dataset("patches_dilated_"+str(style_id), [pointer,patch_size,patch_size,patch_size], np.uint8, compression=9)
        hdf5_file["patches_"+str(style_id)][:] = dict_style_patches[style_id][:pointer]
        hdf5_file["patches_edge_"+str(style_id)][:] = dict_style_patches_edge[style_id][:pointer]
        hdf5_file["patches_dilated_"+str(style_id)][:] = dict_style_patches_dilated[style_id][:pointer]
    hdf5_file.close()




def eval_LP_Div_IOU(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    result_dir = "output_for_eval"
    if not os.path.exists(result_dir):
        print("ERROR: result_dir does not exist! "+result_dir)
        exit(-1)
    
    patches_dir = "unique_patches"
    if not os.path.exists(patches_dir):
        print("ERROR: patches_dir does not exist! "+patches_dir)
        exit(-1)

    output_dir = "eval_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #load style shapes
    fin = open("splits/"+config.data_style+".txt")
    styleset_names = [name.strip() for name in fin.readlines()]
    fin.close()
    styleset_len_original = len(styleset_names)
    styleset_len = min(styleset_len_original, max_num_of_styles)

    #load content shapes
    fin = open("splits/"+config.data_content+".txt")
    dataset_names = [name.strip() for name in fin.readlines()]
    fin.close()
    dataset_len_original = len(dataset_names)
    dataset_len = min(dataset_len_original, max_num_of_contents)



    result_LP_IOU = np.zeros([dataset_len,styleset_len],np.float32)
    result_Div_IOU = np.zeros([dataset_len,styleset_len],np.float32)
    result_Div_IOU_raw = np.zeros([dataset_len,styleset_len,styleset_len],np.int32)


    buffer_size = 256*256*16 #change the buffer size if the input voxel is large
    patch_size = 12
    if not config.asymmetry:
        padding_size = 8 - patch_size//2
    else:
        padding_size = 0
    sample_patch_num = 1000
    IOU_threshold = 0.95

    #prepare dictionary for style shapes
    dict_style_patches_tensor = []
    for style_id in range(styleset_len_original):
        data_dict = h5py.File(patches_dir+"/style_"+str(style_id)+".hdf5", 'r')
        patches = data_dict['patches'][:]
        data_dict.close()
        patches_tensor = torch.from_numpy(patches).to(device)
        dict_style_patches_tensor.append(patches_tensor)

    for content_id in range(dataset_len):
        for style_id in range(styleset_len):

            start_time = time.time()

            voxel_model_file = open(result_dir+"/output_content_"+str(content_id)+"_style_"+str(style_id)+".binvox", 'rb')
            output_shape = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=False).data.astype(np.uint8)
            output_shape = np.ascontiguousarray(output_shape[:,:,padding_size:])

            patches = np.zeros([buffer_size,patch_size,patch_size,patch_size], np.uint8)
            patch_num = cutils.get_patches(output_shape,patches,patch_size)
            if patch_num>sample_patch_num:
                patches = patches[:patch_num]
                np.random.shuffle(patches)
                patches = patches[:sample_patch_num]
                patches = np.ascontiguousarray(patches)
                patch_num = sample_patch_num
            else:
                patches = np.copy(patches[:patch_num])

            this_patches_tensor = torch.from_numpy(patches).to(device)

            #IOU
            similar_flags = np.zeros([patch_num,styleset_len_original], np.int32)
            for patch_id in range(patch_num):
                for compare_id in range(styleset_len_original):
                    patch_tensor = this_patches_tensor[patch_id:patch_id+1]
                    patches_tensor = dict_style_patches_tensor[compare_id]
                    ious = torch.sum( torch.bitwise_and(patch_tensor, patches_tensor), dim=(1,2,3), dtype=torch.int ).float() / torch.sum( torch.bitwise_or(patch_tensor, patches_tensor), dim=(1,2,3), dtype=torch.int ).float()
                    iou = torch.max(ious).item()

                    similar_flags[patch_id,compare_id] = (iou>IOU_threshold)
            Div_IOU_raw = np.sum(similar_flags,axis=0)
            LP_IOU = np.sum(np.max(similar_flags,axis=1))/float(patch_num)

            result_LP_IOU[content_id,style_id] = LP_IOU
            result_Div_IOU_raw[content_id,style_id] = Div_IOU_raw[:styleset_len]

            print(content_id,style_id,time.time()-start_time,LP_IOU)

        #Div
        result_Div_IOU_mean = np.mean(result_Div_IOU_raw.astype(np.float32),axis=1,keepdims=True)
        result_Div_IOU_normalized = result_Div_IOU_raw-result_Div_IOU_mean

        for style_id in range(styleset_len):
            # #top 10%
            # top_N = max(int(0.1*styleset_len),1)
            # ranking = np.argsort(result_Div_IOU_normalized[content_id,style_id])
            # valid_set = ranking[-top_N:]
            # if style_id in valid_set:
            #     Div_IOU = 1
            # else:
            #     Div_IOU = 0
            Div_IOU = (result_Div_IOU_normalized[content_id,style_id,style_id] == np.max(result_Div_IOU_normalized[content_id,style_id]))
            result_Div_IOU[content_id,style_id] = Div_IOU


    #write result_LP_IOU
    fout = open(output_dir+"/result_LP_IOU.txt", 'w')
    for content_id in range(dataset_len):
        for style_id in range(styleset_len):
            fout.write(str(result_LP_IOU[content_id,style_id]))
            if style_id!=styleset_len-1:
                fout.write("\t")
        if content_id!=dataset_len-1:
            fout.write("\n")
    fout.close()
    
    #write result_Div_IOU
    fout = open(output_dir+"/result_Div_IOU.txt", 'w')
    for content_id in range(dataset_len):
        for style_id in range(styleset_len):
            fout.write(str(result_Div_IOU[content_id,style_id]))
            if style_id!=styleset_len-1:
                fout.write("\t")
        if content_id!=dataset_len-1:
            fout.write("\n")
    fout.close()
    
    #write result_Div_IOU_raw
    fout = open(output_dir+"/result_Div_IOU_raw.txt", 'w')
    for content_id in range(dataset_len):
        for style_id in range(styleset_len):
            for compare_id in range(styleset_len):
                fout.write(str(result_Div_IOU_raw[content_id,style_id,compare_id]))
                if style_id!=styleset_len-1 or compare_id!=styleset_len-1:
                    fout.write("\t")
        if content_id!=dataset_len-1:
            fout.write("\n")
    fout.close()
    
    #write result_LP_Div_IOU_mean
    fout = open(output_dir+"/result_LP_Div_IOU_mean.txt", 'w')
    fout.write("LP_IOU:\n"+str(np.mean(result_LP_IOU))+"\n")
    fout.write("Div_IOU:\n"+str(np.mean(result_Div_IOU))+"\n")
    fout.close()



def eval_LP_Div_MAE(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    result_dir = "output_for_eval"
    if not os.path.exists(result_dir):
        print("ERROR: result_dir does not exist! "+result_dir)
        exit(-1)
    
    patches_dir = "unique_patches"
    if not os.path.exists(patches_dir):
        print("ERROR: patches_dir does not exist! "+patches_dir)
        exit(-1)

    output_dir = "eval_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #load style shapes
    fin = open("splits/"+config.data_style+".txt")
    styleset_names = [name.strip() for name in fin.readlines()]
    fin.close()
    styleset_len_original = len(styleset_names)
    styleset_len = min(styleset_len_original, max_num_of_styles)

    #load content shapes
    fin = open("splits/"+config.data_content+".txt")
    dataset_names = [name.strip() for name in fin.readlines()]
    fin.close()
    dataset_len_original = len(dataset_names)
    dataset_len = min(dataset_len_original, max_num_of_contents)



    result_LP_MAE = np.zeros([dataset_len,styleset_len],np.float32)
    result_Div_MAE = np.zeros([dataset_len,styleset_len],np.float32)
    result_Div_MAE_raw = np.zeros([dataset_len,styleset_len,styleset_len],np.int32)


    buffer_size = 256*256*16 #change the buffer size if the input voxel is large
    patch_size = 12
    if not config.asymmetry:
        padding_size = 8 - patch_size//2
    else:
        padding_size = 0
    sample_patch_num = 1000
    MAE_threshold = 0.05
    MAE_threshold_int = int(MAE_threshold*patch_size*patch_size*patch_size)

    #prepare dictionary for style shapes
    dict_style_patches_tensor = []
    for style_id in range(styleset_len_original):
        data_dict = h5py.File(patches_dir+"/style_"+str(style_id)+".hdf5", 'r')
        patches = data_dict['patches'][:]
        data_dict.close()
        patches_tensor = torch.from_numpy(patches).to(device)
        dict_style_patches_tensor.append(patches_tensor)

    for content_id in range(dataset_len):
        for style_id in range(styleset_len):

            start_time = time.time()

            voxel_model_file = open(result_dir+"/output_content_"+str(content_id)+"_style_"+str(style_id)+".binvox", 'rb')
            output_shape = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=False).data.astype(np.uint8)
            output_shape = np.ascontiguousarray(output_shape[:,:,padding_size:])

            patches = np.zeros([buffer_size,patch_size,patch_size,patch_size], np.uint8)
            patch_num = cutils.get_patches(output_shape,patches,patch_size)
            if patch_num>sample_patch_num:
                patches = patches[:patch_num]
                np.random.shuffle(patches)
                patches = patches[:sample_patch_num]
                patches = np.ascontiguousarray(patches)
                patch_num = sample_patch_num
            else:
                patches = np.copy(patches[:patch_num])

            this_patches_tensor = torch.from_numpy(patches).to(device)

            #MAE
            similar_flags = np.zeros([patch_num,styleset_len_original], np.int32)
            for patch_id in range(patch_num):
                for compare_id in range(styleset_len_original):
                    patch_tensor = this_patches_tensor[patch_id:patch_id+1]
                    patches_tensor = dict_style_patches_tensor[compare_id]
                    maes = torch.sum( torch.abs(patch_tensor-patches_tensor), dim=(1,2,3), dtype=torch.int )
                    mae = torch.min(maes).item()

                    similar_flags[patch_id,compare_id] = (mae<=MAE_threshold_int)
            Div_MAE_raw = np.sum(similar_flags,axis=0)
            LP_MAE = np.sum(np.max(similar_flags,axis=1))/float(patch_num)

            result_LP_MAE[content_id,style_id] = LP_MAE
            result_Div_MAE_raw[content_id,style_id] = Div_MAE_raw[:styleset_len]

            print(content_id,style_id,time.time()-start_time,LP_MAE)

        #Div
        result_Div_MAE_mean = np.mean(result_Div_MAE_raw.astype(np.float32),axis=1,keepdims=True)
        result_Div_MAE_normalized = result_Div_MAE_raw-result_Div_MAE_mean

        for style_id in range(styleset_len):
            # #top 10%
            # top_N = max(int(0.1*styleset_len),1)
            # ranking = np.argsort(result_Div_MAE_normalized[content_id,style_id])
            # valid_set = ranking[-top_N:]
            # if style_id in valid_set:
            #     Div_MAE = 1
            # else:
            #     Div_MAE = 0
            Div_MAE = (result_Div_MAE_normalized[content_id,style_id,style_id] == np.max(result_Div_MAE_normalized[content_id,style_id]))
            result_Div_MAE[content_id,style_id] = Div_MAE


    #write result_LP_MAE
    fout = open(output_dir+"/result_LP_MAE.txt", 'w')
    for content_id in range(dataset_len):
        for style_id in range(styleset_len):
            fout.write(str(result_LP_MAE[content_id,style_id]))
            if style_id!=styleset_len-1:
                fout.write("\t")
        if content_id!=dataset_len-1:
            fout.write("\n")
    fout.close()
    
    #write result_Div_MAE
    fout = open(output_dir+"/result_Div_MAE.txt", 'w')
    for content_id in range(dataset_len):
        for style_id in range(styleset_len):
            fout.write(str(result_Div_MAE[content_id,style_id]))
            if style_id!=styleset_len-1:
                fout.write("\t")
        if content_id!=dataset_len-1:
            fout.write("\n")
    fout.close()
    
    #write result_Div_MAE_raw
    fout = open(output_dir+"/result_Div_MAE_raw.txt", 'w')
    for content_id in range(dataset_len):
        for style_id in range(styleset_len):
            for compare_id in range(styleset_len):
                fout.write(str(result_Div_MAE_raw[content_id,style_id,compare_id]))
                if style_id!=styleset_len-1 or compare_id!=styleset_len-1:
                    fout.write("\t")
        if content_id!=dataset_len-1:
            fout.write("\n")
    fout.close()
    
    #write result_LP_Div_MAE_mean
    fout = open(output_dir+"/result_LP_Div_MAE_mean.txt", 'w')
    fout.write("LP_MAE:\n"+str(np.mean(result_LP_MAE))+"\n")
    fout.write("Div_MAE:\n"+str(np.mean(result_Div_MAE))+"\n")
    fout.close()



def eval_LP_Div_Fscore(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    result_dir = "output_for_eval"
    if not os.path.exists(result_dir):
        print("ERROR: result_dir does not exist! "+result_dir)
        exit(-1)
    
    patches_dir = "unique_patches"
    if not os.path.exists(patches_dir):
        print("ERROR: patches_dir does not exist! "+patches_dir)
        exit(-1)

    output_dir = "eval_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #load style shapes
    fin = open("splits/"+config.data_style+".txt")
    styleset_names = [name.strip() for name in fin.readlines()]
    fin.close()
    styleset_len_original = len(styleset_names)
    styleset_len = min(styleset_len_original, max_num_of_styles)

    #load content shapes
    fin = open("splits/"+config.data_content+".txt")
    dataset_names = [name.strip() for name in fin.readlines()]
    fin.close()
    dataset_len_original = len(dataset_names)
    dataset_len = min(dataset_len_original, max_num_of_contents)



    result_LP_Fscore = np.zeros([dataset_len,styleset_len],np.float32)
    result_Div_Fscore = np.zeros([dataset_len,styleset_len],np.float32)
    result_Div_Fscore_raw = np.zeros([dataset_len,styleset_len,styleset_len],np.int32)


    buffer_size = 256*256*16 #change the buffer size if the input voxel is large
    patch_size = 12
    if not config.asymmetry:
        padding_size = 8 - patch_size//2
    else:
        padding_size = 0
    sample_patch_num = 1000
    Fscore_threshold = 0.95

    #prepare dictionary for style shapes
    dict_style_patches_edge_tensor = []
    dict_style_patches_edge_sum_tensor = []
    dict_style_patches_dilated_tensor = []
    for style_id in range(styleset_len_original):
        data_dict = h5py.File(patches_dir+"/style_"+str(style_id)+".hdf5", 'r')
        patches_edge = data_dict['patches_edge'][:]
        patches_dilated = data_dict['patches_dilated'][:]
        data_dict.close()

        patches_edge_tensor = torch.from_numpy(patches_edge).to(device)
        dict_style_patches_edge_tensor.append(patches_edge_tensor)
        dict_style_patches_edge_sum_tensor.append( torch.sum( patches_edge_tensor, dim=(1,2,3), dtype=torch.int ).float() )
        
        patches_dilated_tensor = torch.from_numpy(patches_dilated).to(device)
        dict_style_patches_dilated_tensor.append(patches_dilated_tensor)

    for content_id in range(dataset_len):
        for style_id in range(styleset_len):

            start_time = time.time()

            voxel_model_file = open(result_dir+"/output_content_"+str(content_id)+"_style_"+str(style_id)+".binvox", 'rb')
            output_shape = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=False).data.astype(np.uint8)
            output_shape = np.ascontiguousarray(output_shape[:,:,padding_size:])

            output_shape_tensor = torch.from_numpy(output_shape).to(device).unsqueeze(0).unsqueeze(0).float()
            output_shape_edge_tensor = F.max_pool3d(-output_shape_tensor, kernel_size = 3, stride = 1, padding = 1) + output_shape_tensor
            output_shape_dilated_tensor = F.max_pool3d(output_shape_edge_tensor, kernel_size = 3, stride = 1, padding = 1)
            output_shape_edge = output_shape_edge_tensor.detach().cpu().numpy()[0,0]
            output_shape_edge = np.round(output_shape_edge).astype(np.uint8)
            output_shape_dilated = output_shape_dilated_tensor.detach().cpu().numpy()[0,0]
            output_shape_dilated = np.round(output_shape_dilated).astype(np.uint8)

            patches = np.zeros([buffer_size,patch_size,patch_size,patch_size], np.uint8)
            patches_edge = np.zeros([buffer_size,patch_size,patch_size,patch_size], np.uint8)
            patches_dilated = np.zeros([buffer_size,patch_size,patch_size,patch_size], np.uint8)
            patch_num = cutils.get_patches_edge_dilated(output_shape,output_shape_edge,output_shape_dilated,patches,patches_edge,patches_dilated,patch_size)

            if patch_num>sample_patch_num:
                patch_index_list = np.arange(patch_num)
                np.random.shuffle(patch_index_list)
                patches_edge = patches_edge[patch_index_list[:sample_patch_num]]
                patches_edge = np.ascontiguousarray(patches_edge)
                patches_dilated = patches_dilated[patch_index_list[:sample_patch_num]]
                patches_dilated = np.ascontiguousarray(patches_dilated)
                patch_num = sample_patch_num
            else:
                patches_edge = np.copy(patches_edge[:patch_num])
                patches_dilated = np.copy(patches_dilated[:patch_num])

            this_patches_edge_tensor = torch.from_numpy(patches_edge).to(device)
            this_patches_dilated_tensor = torch.from_numpy(patches_dilated).to(device)

            #Fscore
            similar_flags = np.zeros([patch_num,styleset_len_original], np.int32)
            for patch_id in range(patch_num):
                for compare_id in range(styleset_len_original):

                    patch_edge_tensor = this_patches_edge_tensor[patch_id:patch_id+1]
                    patch_dilated_tensor = this_patches_dilated_tensor[patch_id:patch_id+1]

                    patches_edge_tensor = dict_style_patches_edge_tensor[compare_id]
                    patches_edge_sum_tensor = dict_style_patches_edge_sum_tensor[compare_id]
                    patches_dilated_tensor = dict_style_patches_dilated_tensor[compare_id]

                    precision = torch.sum( torch.bitwise_and(patch_edge_tensor, patches_dilated_tensor), dim=(1,2,3), dtype=torch.int ).float() / torch.sum( patch_edge_tensor, dim=(1,2,3), dtype=torch.int ).float()
                    recall = torch.sum( torch.bitwise_and(patch_dilated_tensor, patches_edge_tensor), dim=(1,2,3), dtype=torch.int ).float() / patches_edge_sum_tensor
                    Fscores = 2*precision*recall/(precision+recall)
                    Fscore = torch.max(Fscores).item()

                    similar_flags[patch_id,compare_id] = (Fscore>Fscore_threshold)
            Div_Fscore_raw = np.sum(similar_flags,axis=0)
            LP_Fscore = np.sum(np.max(similar_flags,axis=1))/float(patch_num)

            result_LP_Fscore[content_id,style_id] = LP_Fscore
            result_Div_Fscore_raw[content_id,style_id] = Div_Fscore_raw[:styleset_len]

            print("eval_LP_Div_Fscore",content_id,style_id,time.time()-start_time,LP_Fscore)

        #Div
        result_Div_Fscore_mean = np.mean(result_Div_Fscore_raw.astype(np.float32),axis=1,keepdims=True)
        result_Div_Fscore_normalized = result_Div_Fscore_raw-result_Div_Fscore_mean

        for style_id in range(styleset_len):
            # #top 10%
            # top_N = max(int(0.1*styleset_len),1)
            # ranking = np.argsort(result_Div_Fscore_normalized[content_id,style_id])
            # valid_set = ranking[-top_N:]
            # if style_id in valid_set:
            #     Div_Fscore = 1
            # else:
            #     Div_Fscore = 0
            Div_Fscore = (result_Div_Fscore_normalized[content_id,style_id,style_id] == np.max(result_Div_Fscore_normalized[content_id,style_id]))
            result_Div_Fscore[content_id,style_id] = Div_Fscore


    #write result_LP_Fscore
    fout = open(output_dir+"/result_LP_Fscore.txt", 'w')
    for content_id in range(dataset_len):
        for style_id in range(styleset_len):
            fout.write(str(result_LP_Fscore[content_id,style_id]))
            if style_id!=styleset_len-1:
                fout.write("\t")
        if content_id!=dataset_len-1:
            fout.write("\n")
    fout.close()
    
    #write result_Div_Fscore
    fout = open(output_dir+"/result_Div_Fscore.txt", 'w')
    for content_id in range(dataset_len):
        for style_id in range(styleset_len):
            fout.write(str(result_Div_Fscore[content_id,style_id]))
            if style_id!=styleset_len-1:
                fout.write("\t")
        if content_id!=dataset_len-1:
            fout.write("\n")
    fout.close()
    
    #write result_Div_Fscore_raw
    fout = open(output_dir+"/result_Div_Fscore_raw.txt", 'w')
    for content_id in range(dataset_len):
        for style_id in range(styleset_len):
            for compare_id in range(styleset_len):
                fout.write(str(result_Div_Fscore_raw[content_id,style_id,compare_id]))
                if style_id!=styleset_len-1 or compare_id!=styleset_len-1:
                    fout.write("\t")
        if content_id!=dataset_len-1:
            fout.write("\n")
    fout.close()
    
    #write result_LP_Div_Fscore_mean
    fout = open(output_dir+"/result_LP_Div_Fscore_mean.txt", 'w')
    fout.write("LP_Fscore:\n"+str(np.mean(result_LP_Fscore))+"\n")
    fout.write("Div_Fscore:\n"+str(np.mean(result_Div_Fscore))+"\n")
    fout.close()



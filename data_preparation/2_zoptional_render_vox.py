import numpy as np
import cv2
import os
import binvox_rw
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("class_id", type=str, help="shapenet category id")
FLAGS = parser.parse_args()

class_id = FLAGS.class_id
target_dir = "./"+class_id+"/"
if not os.path.exists(target_dir):
	print("ERROR: this dir does not exist: "+target_dir)
	exit()

save_dir = "depth_render/"+class_id+"/"
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

obj_names = os.listdir(target_dir)
obj_names = sorted(obj_names)


for i in range(len(obj_names)):
	this_name = target_dir + obj_names[i] + "/model_filled.binvox"
	write_dir1 = target_dir + obj_names[i] + "/depth.png"
	print(i,this_name)


	voxel_model_file = open(this_name, 'rb')
	batch_voxels = binvox_rw.read_as_3d_array(voxel_model_file).data.astype(np.uint8)

	out = np.zeros([512*2,512*4], np.uint8)

	tmp = batch_voxels
	mask = np.amax(tmp, axis=0).astype(np.int32)
	depth = np.argmax(tmp,axis=0)
	depth = 230 + np.clip(np.min(depth+(1-mask)*512) - depth, -180, 0)
	depth = depth*mask
	out[512*0:512*1,512*0:512*1] = depth[::-1,:]

	mask = np.amax(tmp, axis=1).astype(np.int32)
	depth = np.argmax(tmp,axis=1)
	depth = 230 + np.clip(np.min(depth+(1-mask)*512) - depth, -180, 0)
	depth = depth*mask
	out[512*0:512*1,512*1:512*2] = depth

	mask = np.amax(tmp, axis=2).astype(np.int32)
	depth = np.argmax(tmp,axis=2)
	depth = 230 + np.clip(np.min(depth+(1-mask)*512) - depth, -180, 0)
	depth = depth*mask
	out[512*0:512*1,512*2:512*3] = np.transpose(depth)[::-1,::-1]

	tmp = batch_voxels[::-1,:,:]
	mask = np.amax(tmp, axis=0).astype(np.int32)
	depth = np.argmax(tmp,axis=0)
	depth = 230 + np.clip(np.min(depth+(1-mask)*512) - depth, -180, 0)
	depth = depth*mask
	out[512*1:512*2,512*0:512*1] = depth[::-1,::-1]
	redisual = np.clip(np.abs(mask[:,:] - mask[:,::-1])*256,0,255)
	out[512*0:512*1,512*3:512*4] = redisual[::-1,::-1]

	tmp = batch_voxels[:,::-1,:]
	mask = np.amax(tmp, axis=1).astype(np.int32)
	depth = np.argmax(tmp,axis=1)
	depth = 230 + np.clip(np.min(depth+(1-mask)*512) - depth, -180, 0)
	depth = depth*mask
	out[512*1:512*2,512*1:512*2] = depth[:,::-1]
	redisual = np.clip(np.abs(mask[:,:] - mask[:,::-1])*256,0,255)
	out[512*1:512*2,512*3:512*4] = redisual[:,::-1]

	tmp = batch_voxels[:,:,::-1]
	mask = np.amax(tmp, axis=2).astype(np.int32)
	depth = np.argmax(tmp,axis=2)
	depth = 230 + np.clip(np.min(depth+(1-mask)*512) - depth, -180, 0)
	depth = depth*mask
	out[512*1:512*2,512*2:512*3] = np.transpose(depth)[::-1,:]

	cv2.imwrite(write_dir1,out)

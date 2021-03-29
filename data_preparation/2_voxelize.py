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

obj_names = os.listdir(target_dir)
obj_names = sorted(obj_names)


for i in range(len(obj_names)):
    this_name = target_dir + obj_names[i] + "/model.obj"
    print(i,this_name)

    maxx = 0.5
    maxy = 0.5
    maxz = 0.5
    minx = -0.5
    miny = -0.5
    minz = -0.5

    command = "./binvox -bb "+str(minx)+" "+str(miny)+" "+str(minz)+" "+str(maxx)+" "+str(maxy)+" "+str(maxz)+" "+" -d 512 -e "+this_name

    os.system(command)


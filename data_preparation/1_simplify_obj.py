import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("class_id", type=str, help="shapenet category id")
FLAGS = parser.parse_args()

class_id = FLAGS.class_id
source_root = "../ShapeNetCore.v1/"+class_id+"/"
if not os.path.exists(source_root):
    print("ERROR: this dir does not exist: "+source_root)
    exit()

target_dir = "./"+class_id+"/"

obj_names = os.listdir(source_root)
obj_names = sorted(obj_names)


def load_obj(dire):
    fin = open(dire,'r')
    lines = fin.readlines()
    fin.close()
    
    vertices = []
    triangles = []
    
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line)==0:
            continue
        if line[0] == 'v':
            x = float(line[1])
            y = float(line[2])
            z = float(line[3])
            vertices.append([x,y,z])
        if line[0] == 'f':
            x = int(line[1].split("/")[0])
            y = int(line[2].split("/")[0])
            z = int(line[3].split("/")[0])
            triangles.append([x-1,y-1,z-1])
    
    vertices = np.array(vertices, np.float32)
    triangles = np.array(triangles, np.int32)
    
    #remove isolated points
    vertices_mapping = np.full([len(vertices)], -1, np.int32)
    for i in range(len(triangles)):
        for j in range(3):
            vertices_mapping[triangles[i,j]] = 1
    counter = 0
    for i in range(len(vertices)):
        if vertices_mapping[i]>0:
            vertices_mapping[i] = counter
            counter += 1
    vertices = vertices[vertices_mapping>=0]
    triangles = vertices_mapping[triangles]
    
    
    #normalize diagonal=1
    x_max = np.max(vertices[:,0])
    y_max = np.max(vertices[:,1])
    z_max = np.max(vertices[:,2])
    x_min = np.min(vertices[:,0])
    y_min = np.min(vertices[:,1])
    z_min = np.min(vertices[:,2])
    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2
    z_mid = (z_max+z_min)/2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = np.sqrt(x_scale*x_scale + y_scale*y_scale + z_scale*z_scale)
    
    vertices[:,0] = (vertices[:,0]-x_mid)/scale
    vertices[:,1] = (vertices[:,1]-y_mid)/scale
    vertices[:,2] = (vertices[:,2]-z_mid)/scale
    
    return vertices, triangles

def write_obj(dire, vertices, triangles):
    fout = open(dire, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(triangles[ii,0]+1)+" "+str(triangles[ii,1]+1)+" "+str(triangles[ii,2]+1)+"\n")
    fout.close()


for i in range(len(obj_names)):
    this_name = source_root + obj_names[i] + "/model.obj"
    print(i,this_name)
    out_dir = target_dir + obj_names[i]
    out_name = out_dir + "/model.obj"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    v,t = load_obj(this_name)
    write_obj(out_name, v,t)
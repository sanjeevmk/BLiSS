import trimesh
import os
import pickle
import numpy as np
import math
import torch

def read_obj_from_path(full_path):
    _mesh = trimesh.load(full_path,process=False)

    return _mesh

def read_ply_from_path(full_path):
    _mesh = trimesh.load(full_path,process=False)

    return _mesh

def batch_a_pose_vector(pose_vector):
    poses = pose_vector.view(pose_vector.size()[0],24,3)
    axis = np.array([0,0,1])
    angle = math.radians(-64)
    axis_angle = axis*angle
    axis2 = np.array([1,0,0])
    angle2 = math.radians(-4)
    axis_angle2 = axis2*angle2
    poses[:,16,:] = torch.from_numpy(axis_angle) + torch.from_numpy(axis_angle2)

    angle = math.radians(64)
    axis_angle = axis*angle
    axis2 = np.array([1,0,0])
    angle2 = math.radians(-4)
    axis_angle2 = axis2*angle2
    poses[:,17,:] = torch.from_numpy(axis_angle) + torch.from_numpy(axis_angle2)

    return poses.view(pose_vector.size()[0],1,72)

def a_pose_vector(pose_vector):
    poses = pose_vector.view(24,3)
    axis = np.array([0,0,1])
    angle = math.radians(-60)
    axis_angle = axis*angle
    poses[16,:] = torch.from_numpy(axis_angle)

    angle = math.radians(60)
    axis_angle = axis*angle
    poses[17,:] = torch.from_numpy(axis_angle)

    return poses.view(1,72)

def canonical_a_pose_vector(pose_vector):
    poses = pose_vector.view(24,3)
    axis = np.array([0,0,1])
    angle = math.radians(-60)
    axis_angle = axis*angle
    poses[16,:] = torch.from_numpy(axis_angle)

    angle = math.radians(60)
    axis_angle = axis*angle
    poses[17,:] = torch.from_numpy(axis_angle)

    return poses.view(1,72)
    
def read_objs_from_dir(full_path,how_many=None):
    file_names = os.listdir(full_path)
    file_names = list(filter(lambda x:x.endswith('.obj'),file_names))

    if how_many is not None:
        file_names = file_names[:how_many]

    trimesh_meshes = []
    names = []
    index = 0
    for name in file_names:
        _mesh = trimesh.load(os.path.join(full_path,name),process=False)
        trimesh_meshes.append(_mesh)
        names.append(name)

    return trimesh_meshes,names

def vertex_np_array_from_meshes(trimesh_meshes):
    vertices = []
    for _mesh in trimesh_meshes:
        vertices.append(np.array(_mesh.vertices))
    
    return np.array(vertices)

def dump_array_as_pickle(array,dest_path):
    with open(dest_path,'wb') as file:
        pickle.dump(array,file)

def load_array_from_pickle(pickle_path):
    with open(pickle_path,'rb') as file:
        array = pickle.load(file)
    return array 

def normalize_vertices(vertices):
    translation = -1*np.mean(vertices,0)
    vertices += translation

    bmin = np.min(vertices,0)
    bmax = np.max(vertices,0)

    diag = np.sqrt(np.sum((bmax-bmin)**2))
    scale = 1.0/diag

    vertices *= scale

    return vertices,translation,scale

def get_bb_center(vertices):
    bmin = np.min(vertices,0)
    bmax = np.max(vertices,0)

    center = (bmax+bmin)/2.0
    return center

def get_bb_diagonal(vertices):
    bmin = np.min(vertices,0)
    bmax = np.max(vertices,0)

    diag = np.sqrt(np.sum((bmax-bmin)**2))

    return diag

def get_bottom(vertices):
    low_z = np.min(vertices,0)[2]

    return low_z

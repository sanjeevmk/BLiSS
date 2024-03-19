import torch
torch.manual_seed(4)
torch.cuda.manual_seed_all(4)
from omegaconf import OmegaConf
from Optimize import chamfer_register
import os
from DataStar import DataStar
import utils
import logging
import trimesh
from MeshProcessor import MeshProcessor
import numpy as np
import njf as njf
from datetime import datetime,timezone
from Optimize import ProjectionModule
import random
random.seed(4)
from smplpytorch.pytorch.smpl_layer import SMPL_Layer as SMPL
from star import STAR
from pytorch3d.loss.point_mesh_distance import point_face_distance
import matplotlib.cm as cm
import matplotlib

cmap = cm.get_cmap('Reds')
norm = matplotlib.colors.Normalize(vmin=0.0051, vmax=0.03, clip=False)
mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

def find_initial_pose(scan):
    data_star = STAR()
    weights = torch.zeros(1,data_star.num_betas).float().cuda()
    translation = torch.zeros(1,3).float().cuda()
    poses = torch.cuda.FloatTensor(np.zeros((1,72)))


def create_njf_training_data(pca_set_path,njf_set_path,njf_scan_path,njf_train_dir,round,datatype,k,device):
    if not os.path.exists(njf_train_dir):
        os.makedirs(njf_train_dir)

    logger = logging.getLogger("njf_"+datatype+"_data_r"+str(round))

    data_star = DataStar(pca_set_path,device,gender='neutral',k=k,eval=True,round=round)
    #data_star = SMPL()
    scan_files = os.listdir(njf_scan_path)
    gt_files = os.listdir(njf_set_path)
    njf_output_files = os.listdir(njf_set_path)

    index=0
    import csv
    landmarks_dict = OmegaConf.load('landmarks_5.yml')
    landmark_table = []
    for k,v in landmarks_dict.items():
        landmark_table.append([int(v[0]),int(v[1])])
    landmarks = np.array(landmark_table)
    landmarks = torch.from_numpy(landmarks).long().cuda()

    notopt = []
    with open("selected.txt","r") as f:
        notopt = f.read().splitlines()
    notopt = np.array([int(x) for x in notopt])

    for scan_f,gt_f in zip(scan_files,gt_files):
        prefix = scan_f.split(".obj")[0]

        #if "csr1869" not in prefix:
        #    continue

        output_file = ""
        for out_f in njf_output_files:
            if prefix in out_f:
                output_file = out_f

        dest_path = os.path.join(njf_train_dir,prefix+"_output.obj")
        if not os.path.exists(dest_path):
            command = "cp " + os.path.join(njf_set_path,output_file) + " " + dest_path
            os.system(command)
            mesh_processor = MeshProcessor.meshprocessor_from_file(os.path.join(njf_set_path,output_file),torch.float)
            np.save(dest_path.split(".obj")[0]+".npy",mesh_processor.vertices)

        unregistered_mesh = utils.read_obj_from_path(os.path.join(njf_scan_path,scan_f))
        unregistered_vertices = utils.vertex_np_array_from_meshes([unregistered_mesh])

        gt_mesh = utils.read_obj_from_path(os.path.join(njf_set_path,gt_f))
        gt_vertices = utils.vertex_np_array_from_meshes([gt_mesh])
        print(gt_vertices.shape)

        name = scan_f
        unregistered_points = torch.from_numpy(unregistered_vertices).float().to(device)

        if "nl_" in name or "noisy" in name:
            unregistered_points *= 0.001

        poses,weights,translation,shape_T,shape_Posed,aligned_scan = chamfer_register(data_star,unregistered_points,unregistered_mesh.faces,logger,device,landmarks=landmarks)
        poses,weights,translation,shape_T,shape_Posed,aligned_scan = chamfer_register(data_star,unregistered_points,unregistered_mesh.faces,logger,device,poses=poses)
        poses = poses.clone().detach()
        poses[:,60:] = 0.0
        poses[:,30:36] = 0.0
        poses[:,21:27] = 0.0
        shape_Posed = data_star(poses,weights,translation)
        shape_Posed = shape_Posed.squeeze().cpu().detach().numpy()
        #orig_star = SMPL()
        #poses,weights,translation,shape_T,shape_Posed,aligned_scan = chamfer_register(orig_star,unregistered_points,unregistered_mesh.faces,logger,device,landmarks=landmarks) #,weights=weights,given_translation=translation)
        #poses,weights,translation,shape_T,shape_Posed,aligned_scan = chamfer_register(orig_star,unregistered_points,unregistered_mesh.faces,logger,device,poses=poses) #,weights=weights,given_translation=translation)
        _verts = aligned_scan.cpu().detach().numpy()
        color = np.expand_dims(np.array([144.0,210.0,236.0]),0)/255.0
        color = np.repeat(color,_verts.shape[0],axis=0)
        _mesh = trimesh.Trimesh(vertices=_verts,faces=unregistered_mesh.faces,vertex_colors=color,process=False)
        dest_path = os.path.join(njf_train_dir,prefix+"_scan_A.ply")
        _mesh.export(dest_path)
        out_path = os.path.join(njf_train_dir,prefix+"_scan_A.npy")
        np.save(out_path,_verts)

        _verts = shape_T
        _verts  = _verts - np.expand_dims(np.mean(_verts,0),0)
        pred_verts = torch.from_numpy(_verts).float().cuda()
        _mesh = trimesh.Trimesh(vertices=_verts,faces=data_star.faces,process=False)
        dest_path = os.path.join(njf_train_dir,prefix+"_T.ply")
        _mesh.export(dest_path)

        _verts = np.squeeze(gt_vertices)
        _verts  = _verts - np.expand_dims(np.mean(_verts,0),0)
        gt_triangles = torch.from_numpy(_verts[data_star.faces]).float().cuda()
        #_verts[:,1] = _verts[:,1] - np.expand_dims(np.min(_verts,0),0)[:,1] + np.expand_dims(np.min(shape_T,0),0)[:,1]
        zero_index = torch.from_numpy(np.array([0])).long().to(device)
        v2perror = torch.sqrt(point_face_distance(pred_verts,zero_index,gt_triangles,zero_index,len(pred_verts))).cpu().detach().numpy()
        #err = np.mean(np.sqrt(np.sum((_verts-shape_T)**2,axis=1)))
        #print(np.mean(v2perror),v2perror.shape)
        with open(os.path.join(njf_train_dir,prefix+".csv"),"w") as f:
            f.write(str(np.mean(v2perror))+"\n")

        _mesh = trimesh.Trimesh(vertices=_verts,faces=data_star.faces,process=False)
        dest_path = os.path.join(njf_train_dir,prefix+"_GT.ply")
        _mesh.export(dest_path)

        _verts = shape_Posed
        vertex_colors = [mapper.to_rgba(x) for x in v2perror]
        #color = np.expand_dims(np.array([255.0,185.0,185.0])/255.0,0)
        #color = np.repeat(color,_verts.shape[0],axis=0)
        #greycolor = np.expand_dims(np.array([128.0,128.0,128.0])/255.0,0)
        #greycolor = np.repeat(greycolor,notopt.shape[0],axis=0)
        #color[notopt] = greycolor
        #_mesh = trimesh.Trimesh(vertices=_verts,faces=data_star.faces,vertex_colors=vertex_colors,process=False)
        #dest_path = os.path.join(njf_train_dir,prefix+"_posed_err.ply")
        #_mesh.export(dest_path)

        color = np.expand_dims(np.array([255.0,185.0,185.0])/255.0,0)
        color = np.repeat(color,_verts.shape[0],axis=0)
        greycolor = np.expand_dims(np.array([128.0,128.0,128.0])/255.0,0)
        color[notopt] = greycolor
        _mesh = trimesh.Trimesh(vertices=_verts,faces=data_star.faces,vertex_colors=color,process=False)
        dest_path = os.path.join(njf_train_dir,prefix+"_posed.ply")
        _mesh.export(dest_path)

        '''
        mesh_processor = MeshProcessor.meshprocessor_from_array(_verts.astype(np.float64),data_star.faces.cpu().detach().numpy(),torch.float)
        out_path = os.path.join(njf_train_dir,prefix+"_face_normals.npy")
        np.save(out_path,mesh_processor.face_normals)
        out_path = os.path.join(njf_train_dir,prefix+"_vert_normals.npy")
        np.save(out_path,mesh_processor.normals)
        mesh_processor.computeWKS()
        out_path = os.path.join(njf_train_dir,prefix+"_wks.npy")
        np.save(out_path,mesh_processor.faces_wks)
        out_path = os.path.join(njf_train_dir,prefix+"_faces.npy")
        np.save(out_path,mesh_processor.faces)
        out_path = os.path.join(njf_train_dir,prefix+"_input.npy")
        np.save(out_path,mesh_processor.vertices)

        out_path = os.path.join(njf_train_dir,prefix+"_pose.npy")
        np.save(out_path,poses.cpu().detach().numpy())
        out_path = os.path.join(njf_train_dir,prefix+"_weight.npy")
        np.save(out_path,weights.cpu().detach().numpy())
        out_path = os.path.join(njf_train_dir,prefix+"_translation.npy")
        np.save(out_path,translation.cpu().detach().numpy())
        '''
        index+=1

        logger.info("Index: %d Total: %d",index,len(scan_files))

def create_growth_data(pca_set_path,njf_scan_path,random_files,njf_train_dir,round,datatype,device):
    logger = logging.getLogger("njf_"+datatype+"_data_r"+str(round))

    data_star = DataStar(pca_set_path,device,gender='neutral',k=11)
    scan_files = random_files

    test_outdir = njf_train_dir 
    if not os.path.exists(test_outdir):
        os.makedirs(test_outdir)

    index=0
    for scan_f in scan_files:
        prefix = scan_f.split(".obj")[0]

        unregistered_mesh = utils.read_obj_from_path(os.path.join(njf_scan_path,scan_f))
        unregistered_vertices = utils.vertex_np_array_from_meshes([unregistered_mesh])

        name = scan_f
        unregistered_points = torch.from_numpy(unregistered_vertices).float().to(device)

        if "nl_" in name or "noisy" in name:
            unregistered_points *= 0.001

        poses,weights,translation,shape_T,aligned_scan = chamfer_register(data_star,unregistered_points,logger,device)

        _verts = aligned_scan.cpu().detach().numpy()
        _mesh = trimesh.Trimesh(vertices=_verts,faces=unregistered_mesh.faces,process=False)
        dest_path = os.path.join(test_outdir,prefix+"_scan_A.obj")
        _mesh.export(dest_path)
        out_path = os.path.join(test_outdir,prefix+"_scan_A.npy")
        np.save(out_path,_verts)

        _verts = shape_T
        _mesh = trimesh.Trimesh(vertices=_verts,faces=data_star.faces,process=False)
        dest_path = os.path.join(test_outdir,prefix+"_input.obj")
        _mesh.export(dest_path)

        mesh_processor = MeshProcessor.meshprocessor_from_array(_verts.astype(np.float64),data_star.faces.cpu().detach().numpy(),torch.float)
        out_path = os.path.join(test_outdir,prefix+"_face_normals.npy")
        np.save(out_path,mesh_processor.face_normals)
        out_path = os.path.join(test_outdir,prefix+"_vert_normals.npy")
        np.save(out_path,mesh_processor.normals)
        mesh_processor.computeWKS()
        out_path = os.path.join(test_outdir,prefix+"_wks.npy")
        np.save(out_path,mesh_processor.faces_wks)
        out_path = os.path.join(test_outdir,prefix+"_faces.npy")
        np.save(out_path,mesh_processor.faces)
        out_path = os.path.join(test_outdir,prefix+"_input.npy")
        np.save(out_path,mesh_processor.vertices)

        out_path = os.path.join(test_outdir,prefix+"_pose.npy")
        np.save(out_path,poses.cpu().detach().numpy())
        out_path = os.path.join(test_outdir,prefix+"_weight.npy")
        np.save(out_path,weights.cpu().detach().numpy())
        out_path = os.path.join(test_outdir,prefix+"_translation.npy")
        np.save(out_path,translation.cpu().detach().numpy())

        index+=1
        logger.info("Index: %d Total: %d",index,len(scan_files))

if __name__ == "__main__":
    import sys
    torch.multiprocessing.set_start_method('spawn')
    config = OmegaConf.load(sys.argv[1])
    pca_set_path = config['pca_set']
    njf_scan_path = config['njf_set_scans']
    test_scan_path = config['test_set_scans']
    njf_set_path = config['njf_set']
    test_set_path = config['test_set']
    njf_train_dir = config['njf_train_dir']
    num_rounds = config['num_rounds']
    start_round = config['start_round']
    growth_set_src = config['growth_set_src']
    growth_set_rounds = config['growth_set_rounds']
    logdir = config['logdir']
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(os.path.join(njf_train_dir,"train/")):
        os.makedirs(os.path.join(njf_train_dir,"train/"))
    if not os.path.exists(os.path.join(njf_train_dir,"test/")):
        os.makedirs(os.path.join(njf_train_dir,"test/"))

    available_gpus = [torch.device('cuda:'+str(i)) for i in range(torch.cuda.device_count())]
    open(os.path.join(logdir,"logs_eval"),'w')
    logging.basicConfig(filename=os.path.join(logdir,"logs_eval"),
    filemode='a',
    format='%(asctime)s,%(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)

    growth_scans = os.listdir(growth_set_src)
    growth_scans = list(filter(lambda x:x.endswith(".obj"),growth_scans))

    for round in range(start_round,num_rounds):
        create_njf_training_data(pca_set_path,test_set_path,test_scan_path,os.path.join(njf_train_dir,"test/","r"+str(round)+"/"),round,"eval",config['k'],available_gpus[0])
        #njf.eval_no_gt(config,round,available_gpus[0])
        #njf.eval(config,round,available_gpus[0])

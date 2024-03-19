from concurrent.futures import process
import torch
torch.manual_seed(4)
torch.cuda.manual_seed_all(4)
import numpy as np
from omegaconf import OmegaConf
import os
from torch.utils.data import DataLoader
from Dataset import PCA_Pair_Data, PCA_Pair_Data_Chamfer, PCA_Pair_Data_Chamfer_WithObj
import trimesh 
from torch import nn
from networks_local import CentroidNormalWKSFeat, JacobianNetwork, ScanGlobalFeature
from PoissonSystem import MyCuSPLU, poisson_system_matrices_from_mesh
from termcolor import colored
from datetime import datetime,timezone
import matplotlib.cm as cm
from matplotlib.colors import Normalize,LogNorm
from pointnet import PointNetEncoder
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import chamfer_distance
import csv
import sys
sys.path.append("../")
from DataStar import DataStar
import matplotlib.pyplot as plt
import logging
import matplotlib
import utils
from pytorch3d.loss.point_mesh_distance import point_face_distance

cmap = cm.get_cmap('jet')
norm = matplotlib.colors.Normalize(vmin=0.0051, vmax=0.03, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

def get_dataloader(dataset,batch_size):
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=1)
    return dataloader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight)
        #m.bias.data.fill_(0.0)
    if classname.find('Conv1D')!=-1:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

def train(config,round,device,fresh=True):
    logger = logging.getLogger("njf_training_r"+str(round))

    if not os.path.exists(os.path.dirname(config['model_path'])):
        os.makedirs(os.path.dirname(config['model_path']))
    dataset = PCA_Pair_Data_Chamfer(os.path.join(config['njf_train_dir'],"train/"))

    triangle_feature_dim = config['triangle_feature_dim']
    scan_feature_dim = config['scan_feature_dim']
    local_feature_dim = config['local_feature_dim']

    model_path = config['model_path']

    centroid_normal_wks_features_network = PointNetEncoder(channel=106,local_out_dim=triangle_feature_dim,out_dim=triangle_feature_dim).to(device)
    scan_feature_network = PointNetEncoder(channel=3,local_out_dim=local_feature_dim,out_dim=scan_feature_dim).to(device)
    inp_feature_network = PointNetEncoder(channel=3,local_out_dim=local_feature_dim,out_dim=scan_feature_dim).to(device)
    jacobian_network = JacobianNetwork(triangle_feature_dim,scan_feature_dim,local_feature_dim).to(device)
    if fresh:
        centroid_normal_wks_features_network.apply(weights_init)
        scan_feature_network.apply(weights_init)
        inp_feature_network.apply(weights_init)
        jacobian_network.apply(weights_init)
    else:
        centroid_normal_wks_features_network.apply(weights_init)
        scan_feature_network.apply(weights_init)
        inp_feature_network.apply(weights_init)
        jacobian_network.apply(weights_init)
        #centroid_normal_wks_features_network.load_state_dict(torch.load(model_path+'_triangle_r'+str(round)))
        #scan_feature_network.load_state_dict(torch.load(model_path+"_global_r"+str(round)))
        #inp_feature_network.load_state_dict(torch.load(model_path+"_input_r"+str(round)))
        #jacobian_network.load_state_dict(torch.load(model_path+"_jacobian_r"+str(round)))

    params = list(centroid_normal_wks_features_network.parameters()) + list(jacobian_network.parameters()) \
    + list(scan_feature_network.parameters()) + list(inp_feature_network.parameters())
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(params,lr=1e-3)

    best_loss = 1e9
    best_epoch = 0
    for ep in range(config['training']['epochs']):
        mean_dataset_loss = 0.0
        mean_jac_loss = 0.0
        dataset.shuffle()
        for ix,data in enumerate(dataset):
            optimizer.zero_grad()
            inp_vertices,face_normals,vert_normals,wks,out_vertices,inp_triangles,out_triangles,out_scan,faces,names = data
            torch_inp_vertices = torch.from_numpy(inp_vertices).float().to(device)
            inp_triangles = torch.from_numpy(inp_triangles).float().to(device)
            face_normals = torch.from_numpy(face_normals).float().to(device)
            wks = torch.from_numpy(wks).float().to(device)
            vert_normals = torch.from_numpy(vert_normals).float().to(device)
            out_vertices = torch.from_numpy(out_vertices).float().to(device)
            out_vertices = out_vertices - torch.mean(out_vertices,dim=0,keepdim=True)
            out_triangles = torch.from_numpy(out_triangles).float().to(device)
            out_scan = torch.from_numpy(out_scan).float().to(device).unsqueeze(0)
            inp_centroids = torch.mean(inp_triangles,dim=1)

            _,nearest_idx,nearest_points = knn_points(inp_centroids.unsqueeze(0),out_scan,return_nn=True)
            nearest_idx = nearest_idx.squeeze()
            nearest_points = nearest_points.squeeze()
            #nearest_idx = np.random.randint(low=0,high=out_scan.size()[1],size=(faces.shape[0],))
            #nearest_points = out_scan[:,nearest_idx,:].squeeze()

            poisson_matrices = poisson_system_matrices_from_mesh(inp_vertices,faces)
            poisson_solver = poisson_matrices.create_poisson_solver().to(device)
            inp_jacobians = poisson_solver.jacobians_from_vertices(torch_inp_vertices.unsqueeze(0)).contiguous().squeeze().view(faces.shape[0],9)
            input_global_feature,centroid_normal_wks_features = centroid_normal_wks_features_network(torch.cat([inp_centroids,face_normals,wks],-1).unsqueeze(0))
            input_global_feature = input_global_feature.repeat(faces.shape[0],1).squeeze()
            centroid_normal_wks_features = centroid_normal_wks_features.squeeze()

            scan_global_feature,scan_local_features = scan_feature_network(out_scan)
            scan_global_feature = scan_global_feature.repeat(faces.shape[0],1).squeeze()
            #input_global_feature,input_local_features = scan_feature_network(inp_centroids.unsqueeze(0)) 
            #input_global_feature = input_global_feature.repeat(faces.shape[0],1).squeeze()
            #_,nearest_idx,nearest_features = knn_points(input_local_features,scan_local_features,return_nn=True)
            #nearest_idx = nearest_idx.squeeze()
            #nearest_points = out_scan[:,nearest_idx,:].squeeze()
            nearest_features = scan_local_features[:,nearest_idx,:]
            nearest_features = nearest_features.squeeze()


            predicted_jacobians = jacobian_network(centroid_normal_wks_features,input_global_feature,scan_global_feature,nearest_features,nearest_points,inp_jacobians)
            #predicted_jacobians = jacobian_network(centroid_normal_wks_features,input_global_feature,scan_global_feature,inp_jacobians)

            pred_verts = poisson_solver.solve_poisson(predicted_jacobians).squeeze()
            gt_restricted_jacobians = poisson_solver.restricted_jacobians_from_vertices(out_vertices.unsqueeze(0)).contiguous()
            gt_jacobians = poisson_solver.jacobians_from_vertices(out_vertices.unsqueeze(0)).contiguous()
            pred_restricted_jacobians = poisson_solver.restrict_jacobians(predicted_jacobians).contiguous()
            pred_verts = pred_verts - torch.mean(pred_verts,dim=0,keepdim=True)
            map_loss = mse(pred_verts,out_vertices)
            jac_loss = mse(pred_restricted_jacobians,gt_restricted_jacobians)

            loss = 10*map_loss + jac_loss
            loss.backward()
            optimizer.step()

            mean_dataset_loss += map_loss.item()
            mean_jac_loss += jac_loss.item()

            if ix%20==0:
                utc_dt = datetime.now(timezone.utc)
                dt = utc_dt.astimezone()
                hour = dt.hour ; minute = dt.minute ; second = dt.second

                logger.info(colored("Epoch: {0:3d} Loss: {1:2.9f} Best Loss: {2:2.9f} Jac Loss:{3:2.9f} Best Ep: {4:3d} Time: {5:2d}:{6:2d}:{7:02d}"
                .format(ep,mean_dataset_loss/(ix+1),best_loss,mean_jac_loss/(ix+1),best_epoch,hour,minute,second),"blue"))

        mean_dataset_loss /= len(dataset)
        mean_jac_loss /= len(dataset)

        if mean_dataset_loss < best_loss:
            best_loss = mean_dataset_loss
            best_epoch = ep
            torch.save(centroid_normal_wks_features_network.state_dict(),config['model_path']+"_triangle_r"+str(round))
            torch.save(scan_feature_network.state_dict(),config['model_path']+"_global_r"+str(round))
            torch.save(inp_feature_network.state_dict(),config['model_path']+"_input_r"+str(round))
            torch.save(jacobian_network.state_dict(),config['model_path']+"_jacobian_r"+str(round))
        utc_dt = datetime.now(timezone.utc)
        dt = utc_dt.astimezone()
        hour = dt.hour ; minute = dt.minute ; second = dt.second

        logger.info(colored("Epoch: {0:3d} Loss: {1:2.9f} Best Loss: {2:2.9f} Jac Loss:{3:2.9f} Best Ep: {4:3d} Time: {5:2d}:{6:2d}:{7:02d}"
        .format(ep,mean_dataset_loss,best_loss,mean_jac_loss,best_epoch,hour,minute,second),"green"))

    #del centroid_normal_wks_features_network,scan_feature_network,inp_feature_network,jacobian_network
    #torch.cuda.empty_cache()
    #return 0

def plot_histogram(x,out_dir,filename):
    width = np.max(x) - np.min(x)
    bins = np.arange(np.min(x),np.max(x),width/10.0)
    fig = plt.figure()
    plt.hist(x, bins=bins)
    plt.savefig(os.path.join(out_dir,filename))
    plt.close(fig)

def test(config,round,device):
    logger = logging.getLogger("njf_testing_r"+str(round))

    datastar = DataStar(config['pca_set'],device,gender='neutral',k=11)

    dataset = PCA_Pair_Data_Chamfer(os.path.join(config['growth_set_rounds'],"r"+str(round)+"/"),has_output=False)

    triangle_feature_dim = config['triangle_feature_dim']
    scan_feature_dim = config['scan_feature_dim']
    local_feature_dim = config['local_feature_dim']

    centroid_normal_wks_features_network = PointNetEncoder(channel=106,local_out_dim=triangle_feature_dim,out_dim=triangle_feature_dim).to(device)
    centroid_normal_wks_features_network.load_state_dict(torch.load(config['model_path']+'_triangle_r'+str(round)))
    scan_feature_network = PointNetEncoder(channel=3,local_out_dim=local_feature_dim,out_dim=scan_feature_dim).to(device)
    scan_feature_network.load_state_dict(torch.load(config['model_path']+"_global_r"+str(round)))
    inp_feature_network = PointNetEncoder(channel=3,local_out_dim=local_feature_dim,out_dim=scan_feature_dim).to(device)
    inp_feature_network.load_state_dict(torch.load(config['model_path']+"_input_r"+str(round)))
    jacobian_network = JacobianNetwork(triangle_feature_dim,scan_feature_dim,local_feature_dim).to(device)
    jacobian_network.load_state_dict(torch.load(config['model_path']+"_jacobian_r"+str(round)))

    mse = nn.MSELoss()
    error_table = []
    mse_error_table = []
    chamfer_distances = [] 
    mse_distances = []
    for _ in range(1):
        mean_dataset_loss = 0.0
        mean_jac_loss = 0.0
        for ix,data in enumerate(dataset):
            inp_vertices,face_normals,vert_normals,wks,inp_triangles,out_scan,faces,poses,weights,translation,names = data

            torch_inp_vertices = torch.from_numpy(inp_vertices).float().to(device)
            inp_triangles = torch.from_numpy(inp_triangles).float().to(device)
            face_normals = torch.from_numpy(face_normals).float().to(device)
            wks = torch.from_numpy(wks).float().to(device)
            vert_normals = torch.from_numpy(vert_normals).float().to(device)
            out_scan = torch.from_numpy(out_scan).float().to(device).unsqueeze(0)
            inp_centroids = torch.mean(inp_triangles,dim=1)

            poses = torch.from_numpy(poses).float().to(device)
            weights = torch.from_numpy(weights).float().to(device)
            translation = torch.from_numpy(translation).float().to(device)

            _,nearest_idx,nearest_points = knn_points(inp_centroids.unsqueeze(0),out_scan,return_nn=True)
            nearest_idx = nearest_idx.squeeze()
            nearest_points = nearest_points.squeeze()

            #_,nearest_idx,nearest_points = knn_points(inp_centroids.unsqueeze(0),out_scan,return_nn=True)
            #nearest_idx = nearest_idx.squeeze()
            #nearest_points = nearest_points.squeeze()

            #import csv
            #print(out_scan.size())
            #csv.writer(open("scan_test.pts","w"),delimiter=" ").writerows(out_scan.squeeze().cpu().detach().numpy().tolist())
            #exit()

            #_,nearest_idx,nearest_points = knn_points(inp_centroids.unsqueeze(0),out_scan,return_nn=True)
            #nearest_idx = nearest_idx.squeeze()
            #nearest_points = nearest_points.squeeze()

            poisson_matrices = poisson_system_matrices_from_mesh(inp_vertices,faces)
            poisson_solver = poisson_matrices.create_poisson_solver().to(device)
            inp_jacobians = poisson_solver.jacobians_from_vertices(torch_inp_vertices.unsqueeze(0)).contiguous().squeeze().view(faces.shape[0],9)
            input_global_feature,centroid_normal_wks_features = centroid_normal_wks_features_network(torch.cat([inp_centroids,face_normals,wks],-1).unsqueeze(0))
            input_global_feature = input_global_feature.repeat(faces.shape[0],1).squeeze()
            centroid_normal_wks_features = centroid_normal_wks_features.squeeze()

            scan_global_feature,scan_local_features = scan_feature_network(out_scan)
            scan_global_feature = scan_global_feature.repeat(faces.shape[0],1).squeeze()
            #input_global_feature,input_local_features = inp_feature_network(inp_centroids.unsqueeze(0)) 
            #input_global_feature = input_global_feature.repeat(faces.shape[0],1).squeeze()
            nearest_features = scan_local_features[:,nearest_idx,:]
            nearest_features = nearest_features.squeeze()
            #_,nearest_idx,nearest_features = knn_points(input_local_features,scan_local_features,return_nn=True)
            #nearest_features = nearest_features.squeeze()
            #nearest_idx = nearest_idx.squeeze()
            #nearest_points = out_scan[:,nearest_idx,:].squeeze()

            predicted_jacobians = jacobian_network(centroid_normal_wks_features,input_global_feature,scan_global_feature,nearest_features,nearest_points,inp_jacobians)
            #predicted_jacobians = jacobian_network(centroid_normal_wks_features,input_global_feature,scan_global_feature,inp_jacobians)

            pred_verts = poisson_solver.solve_poisson(predicted_jacobians).squeeze()


            np_pred_verts = pred_verts.cpu().detach().numpy()

            pred_verts_a = datastar(poses,weights,translation,v_shaped=pred_verts.unsqueeze(0))
            scan_center = torch.mean(out_scan,dim=1,keepdim=True)
            pred_center = torch.mean(pred_verts_a,dim=1,keepdim=True)
            pred_translation = scan_center - pred_center
            pred_verts_a += pred_translation

            cham_distance,_ = chamfer_distance(pred_verts_a,out_scan)
            chamfer_distances.append(cham_distance.item())

            error_table.append([np_pred_verts,faces,names,str(cham_distance.item())])
            utc_dt = datetime.now(timezone.utc)
            dt = utc_dt.astimezone()
            hour = dt.hour ; minute = dt.minute ; second = dt.second
            logger.info("Index: %d Total: %d Name: %s Chamfer: %f",ix,len(dataset),names,cham_distance)

        chamfer_distances = np.sort(np.array(chamfer_distances))

        utc_dt = datetime.now(timezone.utc)
        dt = utc_dt.astimezone()
        hour = dt.hour ; minute = dt.minute ; second = dt.second

        min_cd = np.min(chamfer_distances)
        std_cd = np.std(chamfer_distances)
        error_table = sorted(error_table,key=lambda x:float(x[3]))
        with open(os.path.join(config['growth_set_rounds'],"r"+str(round)+"/","errors.csv"),"w") as f:
            csv.writer(f,delimiter=" ").writerows(error_table)
        #threshold = 10 #int(0.025*len(error_table))
        threshold = min_cd + std_cd
        threshold_point = len(chamfer_distances[chamfer_distances<threshold])
        error_table = error_table[:threshold_point]
        for row in error_table:
            verts = row[0]
            faces = row[1]
            name = row[2].split("_input.npy")[0]

            _mesh = trimesh.Trimesh(vertices=verts,faces=faces,process=False)
            _mesh.export(os.path.join(config['pca_set'],name+"_r"+str(round)+".obj"))

def eval(config,round,device):
    logger = logging.getLogger("njf_testing_r"+str(round))
    results_path = os.path.join(config['results_dir'],"r_"+str(round)+"/")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    datastar = DataStar(config['pca_set'],device,gender='neutral',k=config['k'],eval=True,round=round)

    dataset = PCA_Pair_Data_Chamfer_WithObj(os.path.join(config['njf_train_dir'],"test/","r"+str(round)+"/"))

    triangle_feature_dim = config['triangle_feature_dim']
    scan_feature_dim = config['scan_feature_dim']
    local_feature_dim = config['local_feature_dim']

    centroid_normal_wks_features_network = PointNetEncoder(channel=106,local_out_dim=triangle_feature_dim,out_dim=triangle_feature_dim).to(device)
    centroid_normal_wks_features_network.load_state_dict(torch.load(config['model_path']+'_triangle_r'+str(round)))
    scan_feature_network = PointNetEncoder(channel=3,local_out_dim=local_feature_dim,out_dim=scan_feature_dim).to(device)
    scan_feature_network.load_state_dict(torch.load(config['model_path']+"_global_r"+str(round)))
    inp_feature_network = PointNetEncoder(channel=3,local_out_dim=local_feature_dim,out_dim=scan_feature_dim).to(device)
    inp_feature_network.load_state_dict(torch.load(config['model_path']+"_input_r"+str(round)))
    jacobian_network = JacobianNetwork(triangle_feature_dim,scan_feature_dim,local_feature_dim).to(device)
    jacobian_network.load_state_dict(torch.load(config['model_path']+"_jacobian_r"+str(round)))

    mse = nn.MSELoss()
    error_table = []
    mse_error_table = []
    l2_error_table = []
    chamfer_distances = [] 
    mse_distances = []
    l2_distances = []
    for _ in range(1):
        mean_dataset_loss = 0.0
        mean_jac_loss = 0.0
        mean_distance = 0.0
        mean_v2p = 0.0
        for ix,data in enumerate(dataset):
            inp_vertices,face_normals,vert_normals,wks,out_vertices,inp_triangles,out_triangles,out_scan,faces,scan_faces,poses,weights,translation,names = data

            torch_inp_vertices = torch.from_numpy(inp_vertices).float().to(device)
            inp_triangles = torch.from_numpy(inp_triangles).float().to(device)
            face_normals = torch.from_numpy(face_normals).float().to(device)
            wks = torch.from_numpy(wks).float().to(device)
            vert_normals = torch.from_numpy(vert_normals).float().to(device)
            out_vertices = torch.from_numpy(out_vertices).float().to(device)
            out_vertices = out_vertices - torch.mean(out_vertices,dim=0,keepdim=True)
            out_triangles = torch.from_numpy(out_triangles).float().to(device)
            out_scan = torch.from_numpy(out_scan).float().to(device).unsqueeze(0)
            inp_centroids = torch.mean(inp_triangles,dim=1)

            poses = torch.from_numpy(poses).float().to(device)
            weights = torch.from_numpy(weights).float().to(device)
            translation = torch.from_numpy(translation).float().to(device)

            a_poses = torch.cuda.FloatTensor(np.zeros((1,72))).to(device)
            a_poses = utils.canonical_a_pose_vector(a_poses)
            #poses,weights,translation = chamfer_register(datastar,out_scan)
            _,nearest_idx,nearest_points = knn_points(inp_centroids.unsqueeze(0),out_scan,return_nn=True)
            nearest_idx = nearest_idx.squeeze()
            nearest_points = nearest_points.squeeze()

            #_,nearest_idx,nearest_points = knn_points(inp_centroids.unsqueeze(0),out_scan,return_nn=True)
            #nearest_idx = nearest_idx.squeeze()
            #nearest_points = nearest_points.squeeze()

            #import csv
            #print(out_scan.size())
            #csv.writer(open("scan_test.pts","w"),delimiter=" ").writerows(out_scan.squeeze().cpu().detach().numpy().tolist())
            #exit()

            #_,nearest_idx,nearest_points = knn_points(inp_centroids.unsqueeze(0),out_scan,return_nn=True)
            #nearest_idx = nearest_idx.squeeze()
            #nearest_points = nearest_points.squeeze()

            poisson_matrices = poisson_system_matrices_from_mesh(inp_vertices,faces)
            poisson_solver = poisson_matrices.create_poisson_solver().to(device)
            inp_jacobians = poisson_solver.jacobians_from_vertices(torch_inp_vertices.unsqueeze(0)).contiguous().squeeze().view(faces.shape[0],9)
            input_global_feature,centroid_normal_wks_features = centroid_normal_wks_features_network(torch.cat([inp_centroids,face_normals,wks],-1).unsqueeze(0))
            input_global_feature = input_global_feature.repeat(faces.shape[0],1).squeeze()
            centroid_normal_wks_features = centroid_normal_wks_features.squeeze()

            scan_global_feature,scan_local_features = scan_feature_network(out_scan)
            scan_global_feature = scan_global_feature.repeat(faces.shape[0],1).squeeze()
            #input_global_feature,input_local_features = inp_feature_network(inp_centroids.unsqueeze(0)) 
            #input_global_feature = input_global_feature.repeat(faces.shape[0],1).seeze()
            nearest_features = scan_local_features[:,nearest_idx,:]
            nearest_features = nearest_features.squeeze()
            #_,nearest_idx,nearest_features = knn_points(input_local_features,scan_local_features,return_nn=True)
            #nearest_features = nearest_features.squeeze()
            #nearest_idx = nearest_idx.squeeze()
            #nearest_points = out_scan[:,nearest_idx,:].squeeze()

            predicted_jacobians = jacobian_network(centroid_normal_wks_features,input_global_feature,scan_global_feature,nearest_features,nearest_points,inp_jacobians)
            #predicted_jacobians = jacobian_network(centroid_normal_wks_features,input_global_feature,scan_global_feature,inp_jacobians)

            pred_verts = poisson_solver.solve_poisson(predicted_jacobians).squeeze()
            gt_restricted_jacobians = poisson_solver.restricted_jacobians_from_vertices(out_vertices.unsqueeze(0)).contiguous()
            pred_restricted_jacobians = poisson_solver.restrict_jacobians(predicted_jacobians).contiguous()
            pred_verts = pred_verts - torch.mean(pred_verts,dim=0,keepdim=True)

            map_loss = mse(pred_verts,out_vertices)
            jac_loss = mse(pred_restricted_jacobians,gt_restricted_jacobians)

            per_vertex_error_inp = torch.sqrt(torch.sum(torch.pow(torch_inp_vertices - out_vertices,2),dim=1)).cpu().detach().numpy()
            high_inp = np.max(per_vertex_error_inp)
            per_vertex_error_pred = torch.sqrt(torch.sum(torch.pow(pred_verts - out_vertices,2),dim=1)).cpu().detach().numpy()
            high_pred = np.max(per_vertex_error_pred)
            zero_index = torch.from_numpy(np.array([0])).long().to(device)
            v2perror = torch.mean(torch.sqrt(point_face_distance(pred_verts,zero_index,out_triangles,zero_index,len(out_vertices))))
            logger.info("%s %s %s",names,np.mean(per_vertex_error_inp),np.mean(per_vertex_error_pred))
            #vertex_colors = cmap(10*per_vertex_error_pred)[:,:3]
            vertex_colors = [mapper.to_rgba(x) for x in per_vertex_error_pred]
            pred_vertex_colors = list(vertex_colors)
            np_pred_verts = pred_verts.cpu().detach().numpy()
            _mesh = trimesh.Trimesh(vertices=np_pred_verts,faces=faces,process=False,vertex_colors=vertex_colors)
            _mesh.export(os.path.join(results_path,names.split("_input")[0]+"_pred.ply"))
            np_gt_verts = out_vertices.cpu().detach().numpy()
            _mesh = trimesh.Trimesh(vertices=np_gt_verts,faces=faces,process=False)
            _mesh.export(os.path.join(results_path,names.split("_input")[0]+"_gt.ply"))
            #vertex_colors = cmap(10*per_vertex_error_inp)[:,:3]
            vertex_colors = [mapper.to_rgba(x) for x in per_vertex_error_inp]
            cyan_color = [[0,0.807,0.819] for x in range(out_scan.size()[1])]
            _mesh = trimesh.Trimesh(vertices=inp_vertices,faces=faces,process=False,vertex_colors=vertex_colors)
            _mesh.export(os.path.join(results_path,names.split(".npy")[0]+".ply"))
            _mesh = trimesh.Trimesh(vertices=out_scan.squeeze().cpu().detach().numpy(),faces=scan_faces,process=False,vertex_colors=cyan_color)
            _mesh.export(os.path.join(results_path,names.split(".npy")[0]+"_scan.ply"))

            inp_verts_a = datastar(a_poses,weights,translation,v_shaped=torch_inp_vertices.unsqueeze(0))
            pred_verts_a = datastar(poses,weights,translation,v_shaped=pred_verts.unsqueeze(0))
            scan_center = torch.mean(out_scan,dim=1,keepdim=True)
            pred_center = torch.mean(pred_verts_a,dim=1,keepdim=True)
            pred_translation = scan_center - pred_center
            pred_verts_a += pred_translation
            inp_verts_a += pred_translation
            np_pred_verts_a = pred_verts_a.squeeze().cpu().detach().numpy()
            np_inp_verts_a = inp_verts_a.squeeze().cpu().detach().numpy()
            _mesh = trimesh.Trimesh(vertices=np_pred_verts_a,faces=faces,process=False) #,vertex_colors=vertex_colors)
            _mesh.export(os.path.join(results_path,names.split("_input")[0]+"_pred_a.ply"))
            _mesh = trimesh.Trimesh(vertices=np_inp_verts_a,faces=faces,process=False,vertex_colors=pred_vertex_colors)
            _mesh.export(os.path.join(results_path,names.split("_input")[0]+"_inp_a.ply"))

            cham_distance,_ = chamfer_distance(pred_verts_a,out_scan)
            chamfer_distances.append(cham_distance.item())
            mse_distances.append(map_loss.item())
            l2_distances.append(np.mean(per_vertex_error_pred))
            mean_distance += np.mean(per_vertex_error_pred)

            mean_v2p += v2perror
            mean_dataset_loss += map_loss.item()
            mean_jac_loss += jac_loss.item()

            error_table.append([np_pred_verts,faces,names,str(cham_distance.item())])
            mse_error_table.append([names,str(map_loss.item())])
            l2_error_table.append([names,str(np.mean(per_vertex_error_pred))])
            utc_dt = datetime.now(timezone.utc)
            dt = utc_dt.astimezone()
            hour = dt.hour ; minute = dt.minute ; second = dt.second
            logger.info("Index: %d Total: %d Name: %s Chamfer: %f Distance: %f v2p: %f",ix,len(dataset),names,cham_distance,mean_distance/(ix+1),mean_v2p/(ix+1))
            logger.info(colored("Loss: {0:2.9f} Jac Loss:{1:2.9f} Time: {2:2d}:{3:2d}:{4:02d}"
            .format(mean_dataset_loss/(ix+1),mean_jac_loss/(ix+1),hour,minute,second),"blue"))

        mean_dataset_loss /= len(dataset)
        mean_jac_loss /= len(dataset)
        chamfer_distances = np.array(chamfer_distances)
        mse_distances = np.array(mse_distances)
        l2_distances = np.array(l2_distances)
        plot_histogram(chamfer_distances,results_path,"r"+str(round)+"_chamfer.png")
        plot_histogram(mse_distances,results_path,"r"+str(round)+"_mse.png")
        plot_histogram(l2_distances,results_path,"r"+str(round)+"_l2.png")

        utc_dt = datetime.now(timezone.utc)
        dt = utc_dt.astimezone()
        hour = dt.hour ; minute = dt.minute ; second = dt.second

        error_table = sorted(error_table,key=lambda x:float(x[3]))
        with open(os.path.join(results_path,"errors.csv"),"w") as f:
            csv.writer(f,delimiter=" ").writerows(error_table)

        mse_error_table = sorted(mse_error_table,key=lambda x:float(x[1]))
        with open(os.path.join(results_path,"mse_errors.csv"),"w") as f:
            csv.writer(f,delimiter=" ").writerows(mse_error_table)

        l2_error_table = sorted(l2_error_table,key=lambda x:float(x[1]))
        with open(os.path.join(results_path,"l2_errors.csv"),"w") as f:
            csv.writer(f,delimiter=" ").writerows(l2_error_table)

        logger.info(colored("Loss: {0:2.9f} Jac Loss:{1:2.9f} Time: {2:2d}:{3:2d}:{4:02d}"
        .format(mean_dataset_loss,mean_jac_loss,hour,minute,second),"green"))

def eval_no_gt(config,round,device):
    logger = logging.getLogger("njf_testing_r"+str(round))
    results_path = os.path.join(config['results_dir'],"r_"+str(round)+"/")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    datastar = DataStar(config['pca_set'],device,gender='neutral',k=config['k'],eval=True,round=round)

    dataset = PCA_Pair_Data_Chamfer_WithObj(os.path.join(config['njf_train_dir'],"test/","r"+str(round)+"/"))

    triangle_feature_dim = config['triangle_feature_dim']
    scan_feature_dim = config['scan_feature_dim']
    local_feature_dim = config['local_feature_dim']

    centroid_normal_wks_features_network = PointNetEncoder(channel=106,local_out_dim=triangle_feature_dim,out_dim=triangle_feature_dim).to(device)
    centroid_normal_wks_features_network.load_state_dict(torch.load(config['model_path']+'_triangle_r'+str(round)))
    scan_feature_network = PointNetEncoder(channel=3,local_out_dim=local_feature_dim,out_dim=scan_feature_dim).to(device)
    scan_feature_network.load_state_dict(torch.load(config['model_path']+"_global_r"+str(round)))
    inp_feature_network = PointNetEncoder(channel=3,local_out_dim=local_feature_dim,out_dim=scan_feature_dim).to(device)
    inp_feature_network.load_state_dict(torch.load(config['model_path']+"_input_r"+str(round)))
    jacobian_network = JacobianNetwork(triangle_feature_dim,scan_feature_dim,local_feature_dim).to(device)
    jacobian_network.load_state_dict(torch.load(config['model_path']+"_jacobian_r"+str(round)))

    mse = nn.MSELoss()
    error_table = []
    mse_error_table = []
    l2_error_table = []
    chamfer_distances = [] 
    mse_distances = []
    l2_distances = []
    for _ in range(1):
        mean_dataset_loss = 0.0
        mean_jac_loss = 0.0
        mean_distance = 0.0
        for ix,data in enumerate(dataset):
            inp_vertices,face_normals,vert_normals,wks,out_vertices,inp_triangles,out_triangles,out_scan,faces,scan_faces,poses,weights,translation,names = data

            torch_inp_vertices = torch.from_numpy(inp_vertices).float().to(device)
            inp_triangles = torch.from_numpy(inp_triangles).float().to(device)
            face_normals = torch.from_numpy(face_normals).float().to(device)
            wks = torch.from_numpy(wks).float().to(device)
            vert_normals = torch.from_numpy(vert_normals).float().to(device)
            out_vertices = torch.from_numpy(out_vertices).float().to(device)
            out_vertices = out_vertices - torch.mean(out_vertices,dim=0,keepdim=True)
            out_triangles = torch.from_numpy(out_triangles).float().to(device)
            out_scan = torch.from_numpy(out_scan).float().to(device).unsqueeze(0)
            inp_centroids = torch.mean(inp_triangles,dim=1)

            poses = torch.from_numpy(poses).float().to(device)
            weights = torch.from_numpy(weights).float().to(device)
            translation = torch.from_numpy(translation).float().to(device)

            a_poses = torch.cuda.FloatTensor(np.zeros((1,72))).to(device)
            a_poses = utils.canonical_a_pose_vector(a_poses)
            #poses,weights,translation = chamfer_register(datastar,out_scan)
            _,nearest_idx,nearest_points = knn_points(inp_centroids.unsqueeze(0),out_scan,return_nn=True)
            nearest_idx = nearest_idx.squeeze()
            nearest_points = nearest_points.squeeze()

            #_,nearest_idx,nearest_points = knn_points(inp_centroids.unsqueeze(0),out_scan,return_nn=True)
            #nearest_idx = nearest_idx.squeeze()
            #nearest_points = nearest_points.squeeze()

            #import csv
            #print(out_scan.size())
            #csv.writer(open("scan_test.pts","w"),delimiter=" ").writerows(out_scan.squeeze().cpu().detach().numpy().tolist())
            #exit()

            #_,nearest_idx,nearest_points = knn_points(inp_centroids.unsqueeze(0),out_scan,return_nn=True)
            #nearest_idx = nearest_idx.squeeze()
            #nearest_points = nearest_points.squeeze()

            poisson_matrices = poisson_system_matrices_from_mesh(inp_vertices,faces)
            poisson_solver = poisson_matrices.create_poisson_solver().to(device)
            inp_jacobians = poisson_solver.jacobians_from_vertices(torch_inp_vertices.unsqueeze(0)).contiguous().squeeze().view(faces.shape[0],9)
            input_global_feature,centroid_normal_wks_features = centroid_normal_wks_features_network(torch.cat([inp_centroids,face_normals,wks],-1).unsqueeze(0))
            input_global_feature = input_global_feature.repeat(faces.shape[0],1).squeeze()
            centroid_normal_wks_features = centroid_normal_wks_features.squeeze()

            scan_global_feature,scan_local_features = scan_feature_network(out_scan)
            scan_global_feature = scan_global_feature.repeat(faces.shape[0],1).squeeze()
            #input_global_feature,input_local_features = inp_feature_network(inp_centroids.unsqueeze(0)) 
            #input_global_feature = input_global_feature.repeat(faces.shape[0],1).seeze()
            nearest_features = scan_local_features[:,nearest_idx,:]
            nearest_features = nearest_features.squeeze()
            #_,nearest_idx,nearest_features = knn_points(input_local_features,scan_local_features,return_nn=True)
            #nearest_features = nearest_features.squeeze()
            #nearest_idx = nearest_idx.squeeze()
            #nearest_points = out_scan[:,nearest_idx,:].squeeze()

            predicted_jacobians = jacobian_network(centroid_normal_wks_features,input_global_feature,scan_global_feature,nearest_features,nearest_points,inp_jacobians)
            #predicted_jacobians = jacobian_network(centroid_normal_wks_features,input_global_feature,scan_global_feature,inp_jacobians)

            pred_verts = poisson_solver.solve_poisson(predicted_jacobians).squeeze()
            pred_verts = pred_verts - torch.mean(pred_verts,dim=0,keepdim=True)
            other_color = [[214.0/255.0,146.0/255.0,148.0/255.0] for x in range(pred_verts.size()[0])]
            cyan_color = [[0,0.807,0.819] for x in range(out_scan.size()[1])]
            np_pred_verts = pred_verts.cpu().detach().numpy()
            _mesh = trimesh.Trimesh(vertices=np_pred_verts,faces=faces,process=False,vertex_colors=other_color)
            _mesh.export(os.path.join(results_path,names.split("_input")[0]+"_pred.ply"))
            np_gt_verts = out_vertices.cpu().detach().numpy()
            _mesh = trimesh.Trimesh(vertices=out_scan.squeeze().cpu().detach().numpy(),faces=scan_faces,process=False,vertex_colors=cyan_color)
            _mesh.export(os.path.join(results_path,names.split("_input")[0]+"_scan.ply"))

            inp_verts_a = datastar(a_poses,weights,translation,v_shaped=torch_inp_vertices.unsqueeze(0))
            pred_verts_a = datastar(poses,weights,translation,v_shaped=pred_verts.unsqueeze(0))
            scan_center = torch.mean(out_scan,dim=1,keepdim=True)
            pred_center = torch.mean(pred_verts_a,dim=1,keepdim=True)
            pred_translation = scan_center - pred_center
            pred_verts_a += pred_translation
            inp_verts_a += pred_translation
            np_pred_verts_a = pred_verts_a.squeeze().cpu().detach().numpy()
            np_inp_verts_a = inp_verts_a.squeeze().cpu().detach().numpy()
            _mesh = trimesh.Trimesh(vertices=np_pred_verts_a,faces=faces,process=False) #,vertex_colors=vertex_colors)
            _mesh.export(os.path.join(results_path,names.split("_input")[0]+"_pred_a.ply"))
            _mesh = trimesh.Trimesh(vertices=np_inp_verts_a,faces=faces,process=False,vertex_colors=other_color)
            _mesh.export(os.path.join(results_path,names.split("_input")[0]+"_inp_a.ply"))

            utc_dt = datetime.now(timezone.utc)
            dt = utc_dt.astimezone()
            hour = dt.hour ; minute = dt.minute ; second = dt.second
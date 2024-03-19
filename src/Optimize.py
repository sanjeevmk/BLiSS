import torch
torch.manual_seed(4)
torch.cuda.manual_seed_all(4)
import torch.nn as nn
import numpy as np
from pytorch3d.loss import chamfer_distance
import trimesh
import utils

class ProjectionModule(nn.Module):
    def __init__(self,star_module,device,poses=None,weights=None,translation=None,landmarks=None,not_hands=None):
        super().__init__()
        if weights is None:
            if landmarks is None:
                self.weights = torch.nn.Parameter(torch.zeros(1,star_module.num_betas).float().to(device))
            else:
                self.weights = torch.zeros(1,star_module.num_betas).float().to(device)
                #self.weights = torch.nn.Parameter(torch.zeros(1,star_module.num_betas).float().to(device))
        else:
            self.weights = weights.clone().detach().requires_grad_(False)
        if translation is None:
            if landmarks is None:
                self.translation = torch.nn.Parameter(torch.zeros(1,3).float().to(device))
            else:
                #self.translation = torch.nn.Parameter(torch.zeros(1,3).float().to(device))
                self.translation = torch.zeros(1,3).float().to(device)
        else:
            self.translation = translation.clone().detach().requires_grad_(False)
        if poses is None:
            poses = torch.cuda.FloatTensor(np.zeros((1,72)))
            poses = utils.a_pose_vector(poses)
            self.poses = torch.nn.Parameter(poses.float().to(device))
        else:
            self.poses = torch.nn.Parameter(poses.clone().detach())

        self.basis = star_module.shapedirs.view(star_module.v_template.size()[0]*star_module.v_template.size()[1],-1)
        self.mean = star_module.v_template
        self.mse = nn.MSELoss()
        self.star_module = star_module
        self.v_shaped = None
        self.faces = star_module.faces
        self.zero_index = torch.from_numpy(np.array([0])).long().to(device)
        self.landmarks = landmarks
        self.not_hands = not_hands
        #self.mask = torch.ones(self.poses.size()).float().cuda()
        #self.mask[0,60:] = 0
        #self.mask[0,30:36] = 0
        #self.mask[0,21:27] = 0

    def chamfer_forward(self,points):
        if self.landmarks is None:
            reconstruction = self.star_module(self.poses,self.weights,self.translation,self.v_shaped)
            distance,_ = chamfer_distance(reconstruction,points.unsqueeze(0))
        else:
            reconstruction = self.star_module(self.poses,self.weights,self.translation,self.v_shaped)
            print(reconstruction.squeeze()[self.landmarks[:,0],:])
            print(points[self.landmarks[:,1],:])
            print()
            distance = self.mse(reconstruction.squeeze()[self.landmarks[:,0],:],points[self.landmarks[:,1],:])
        return distance,reconstruction,reconstruction.v_shaped

    def set_v_shaped(self,v_shaped):
        self.v_shaped = torch.nn.Parameter(v_shaped.clone().detach())

def chamfer_register(data_star,scan,scan_faces,logger,device,poses=None,weights=None,given_translation=None,landmarks=None,not_hands=None):
    data_star.v_template = data_star.v_template.squeeze()
    reg_center= torch.mean(data_star.v_template,0,keepdim=True)
    U,S,unregistered_axes = torch.pca_lowrank(scan[0])
    U,S,registered_axes = torch.pca_lowrank(data_star.v_template)
    unregistered_axes[:,2] *= -1
    R = registered_axes.matmul(torch.linalg.inv(unregistered_axes))
    #aligned_points = scan[0].matmul(R.transpose(0,1))
    aligned_points = scan[0]
    unreg_center = torch.mean(aligned_points,0,keepdim=True)
    #translation = reg_center - unreg_center
    #aligned_points -= unreg_center 
    data_star.v_template = data_star.v_template - reg_center + unreg_center
    reg_min,_ = torch.min(data_star.v_template,0,keepdim=True)
    unreg_min,_ = torch.min(aligned_points,0,keepdim=True)
    data_star.v_template[:,1] = data_star.v_template[:,1] - reg_min[:,1] + unreg_min[:,1]
    _mesh = trimesh.Trimesh(vertices=aligned_points.cpu().detach().numpy(),faces=scan_faces,process=False)
    _mesh.export("scan.obj")
    _mesh = trimesh.Trimesh(vertices=data_star.v_template.cpu().detach().numpy(),faces=data_star.faces,process=False)
    _mesh.export("template.obj")
    #exit()


    projection_module = ProjectionModule(data_star,device,poses=poses,weights=weights,translation=given_translation,landmarks=landmarks,not_hands=not_hands)
    optimizer = torch.optim.Adam(projection_module.parameters(),lr=1e-2)

    best_recon_error = np.Inf
    best_poses = []
    best_weights = []
    best_translation = []
    best_vertices_6890_T = []
    best_vertices_6890_Posed = []

    num_plateau = 0
    mean_gradient = 0
    previous_error = 1e9
    for i in range(3500):
        optimizer.zero_grad()
        reconstruction_error,reconstruction,reconstruction_shaped = projection_module.chamfer_forward(aligned_points.squeeze())
        reconstruction_error.backward()
        #mean_gradient = torch.mean(projection_module.poses.grad).item()/3.0
        #mean_gradient += torch.mean(projection_module.weights.grad).item()/3.0
        #mean_gradient += torch.mean(projection_module.translation.grad).item()/3.0
        #mean_gradient /= (i+1)
        optimizer.step()

        if reconstruction_error < best_recon_error:
            best_recon_error = reconstruction_error
            best_poses = projection_module.poses
            best_weights = projection_module.weights
            best_translation = projection_module.translation
            reconstruction_shaped_trans = reconstruction_shaped + best_translation.unsqueeze(0)
            best_vertices_6890_T = reconstruction_shaped_trans.squeeze().cpu().detach().numpy()
            reconstruction_shaped_trans = reconstruction
            best_vertices_6890_Posed = reconstruction_shaped_trans.squeeze().cpu().detach().numpy()

        #if abs(mean_gradient) < 1e-6:
        if abs(reconstruction_error.item() - previous_error)<1e-10:
            logger.info("Recon error: %f Best Error: %f Ep: %d",reconstruction_error.item(),best_recon_error.item(),i)
            return best_poses,best_weights,best_translation,best_vertices_6890_T,best_vertices_6890_Posed,aligned_points
        previous_error = reconstruction_error.item()
        if i%100==0:
            logger.info("Recon error: %f Best Error: %f Gradient: %.15f Ep: %d",reconstruction_error.item(),best_recon_error.item(),mean_gradient,i)
    return best_poses,best_weights,best_translation,best_vertices_6890_T,best_vertices_6890_Posed,aligned_points

def chamfer_register_faust(data_star,scan,scan_faces,logger,device):
    unreg_center = torch.mean(scan,1,keepdim=True)
    reg_center= torch.mean(data_star.v_template,0,keepdim=True)
    translation = reg_center - unreg_center
    scan += translation

    #U,S,unregistered_axes = torch.pca_lowrank(scan[0])
    #U,S,registered_axes = torch.pca_lowrank(data_star.v_template)
    #unregistered_axes[:,2] *= -1
    #R = registered_axes.matmul(torch.linalg.inv(unregistered_axes))
    #aligned_points = scan[0].matmul(R.transpose(0,1))
    aligned_points = scan

    projection_module = ProjectionModule(data_star,device)
    optimizer = torch.optim.Adam(projection_module.parameters(),lr=1e-2)

    best_recon_error = np.Inf
    best_poses = []
    best_weights = []
    best_translation = []
    best_vertices_6890_T = []
    best_vertices = []

    num_plateau = 0
    mean_gradient = 0
    previous_error = 1e9
    zero_index = torch.from_numpy(np.array([0])).long().to(device)
    for i in range(2500):
        optimizer.zero_grad()
        reconstruction_error,reconstruction,reconstruction_shaped = projection_module.chamfer_forward(aligned_points.squeeze())
        reconstruction_error.backward()
        mean_gradient = torch.mean(projection_module.poses.grad).item()/3.0
        mean_gradient += torch.mean(projection_module.weights.grad).item()/3.0
        mean_gradient += torch.mean(projection_module.translation.grad).item()/3.0
        #mean_gradient /= (i+1)
        optimizer.step()

        if reconstruction_error < best_recon_error:
            best_recon_error = reconstruction_error
            best_poses = projection_module.poses
            best_weights = projection_module.weights
            best_translation = projection_module.translation
            reconstruction_shaped_trans = reconstruction_shaped + best_translation.unsqueeze(0)
            best_vertices_6890_T = reconstruction_shaped_trans.squeeze().cpu().detach().numpy()
            best_vertices = reconstruction.squeeze().cpu().detach().numpy()

        #if abs(mean_gradient) < 1e-6:
        if abs(reconstruction_error.item() - previous_error)<1e-10:
            logger.info("Recon error: %f Best Error: %f Ep: %d",reconstruction_error.item(),best_recon_error.item(),i)
            return best_poses,best_weights,best_translation,best_vertices_6890_T,aligned_points,best_vertices
        previous_error = reconstruction_error.item()
        if i%100==0:
            logger.info("Recon error: %f Best Error: %f Gradient: %.15f Ep: %d",reconstruction_error.item(),best_recon_error.item(),mean_gradient,i)

    logger.info("Recon error: %f Best Error: %f Ep: %d",reconstruction_error.item(),best_recon_error.item(),i)
    return best_poses,best_weights,best_translation,best_vertices_6890_T,aligned_points,best_vertices
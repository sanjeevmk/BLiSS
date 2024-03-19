import torch
import torch.nn as nn
import numpy as np
import os 
try:
    import cPickle as pickle
except ImportError:
    import pickle
from star_utils import rodrigues , quat_feat , with_zeros
from star_config import cfg 
import utils
from learn_pca_basis import get_pca_basis_from_np_vertices
from torch.nn import MSELoss

class DataStar(nn.Module):
    def __init__(self,path,device,k=None,gender='female',cache="./cached_meshes",eval=False,round=0):
        super(DataStar, self).__init__()
        if gender not in ['male','female','neutral']:
            raise RuntimeError('Invalid Gender')

        if gender == 'male':
            path_model = cfg.path_male_star
        elif gender == 'female':
            path_model = cfg.path_female_star
        else:
            path_model = cfg.path_neutral_star

        if not os.path.exists(path_model):
            raise RuntimeError('Path does not exist %s' % (path_model))
        import numpy as np
        self.device = device

        star_model = np.load(path_model,allow_pickle=True)
        J_regressor = star_model['J_regressor']
        rows,cols = np.where(J_regressor!=0)
        vals = J_regressor[rows,cols]
        self.num_betas = k 

        # Model sparse joints regressor, regresses joints location from a mesh
        self.register_buffer('J_regressor', torch.FloatTensor(J_regressor).to(device))

        # Model skinning weights
        self.register_buffer('weights', torch.FloatTensor(star_model['weights']).to(device))
        # Model pose corrective blend shapes
        self.register_buffer('posedirs', torch.FloatTensor(star_model['posedirs'].reshape((-1,93))).to(device))
        # Mean Shape
        self.register_buffer('v_template_star', torch.FloatTensor(star_model['v_template']).to(device))
        # Shape corrective blend shapes
        self.register_buffer('shapedirs_star', torch.FloatTensor(np.array(star_model['shapedirs'][:,:,:k])).to(device))
        # Mesh traingles
        self.register_buffer('faces', torch.from_numpy(star_model['f'].astype(np.int64)).to(device))
        self.f = star_model['f']
        # Kinematic tree of the model
        self.register_buffer('kintree_table', torch.from_numpy(star_model['kintree_table'].astype(np.int64)).to(device))

        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.register_buffer('parent', torch.LongTensor(
            [id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])]))

        self.verts = None
        self.J = None
        self.R = None

        if not eval:
            scale,basis,mean_shape = self.get_custom_basis_from_dataset(path,k=k,cache=cache)
        else:
            scale,basis,mean_shape = self.get_round_basis_from_dataset(path,round,k=k,cache=cache)
        #np.save("our_basis_smplify.npz",basis)
        #print(basis.shape)
        #exit()
        self.register_buffer('v_template', torch.FloatTensor(mean_shape).to(device))
        # Shape corrective blend shapes
        self.register_buffer('shapedirs', torch.FloatTensor(basis).to(device))
        print(self.shapedirs.size(),self.num_betas)


    def load_dataset(self,path,cache="./cached_meshes"):
        if os.path.isdir(path) :
            registered_meshes,names = utils.read_objs_from_dir(path)
            vertices = utils.vertex_np_array_from_meshes(registered_meshes)
            faces = registered_meshes[0].faces
            vertices_for_mean = np.array([vertices[i] for i in range(len(vertices)) if "Wrapped" in names[i]])
            if not os.path.exists(cache):
                os.makedirs(cache)
            utils.dump_array_as_pickle(registered_meshes,os.path.join(cache,"meshes.pkl"))
        else:
            meshes = utils.load_array_from_pickle(path)
            vertices = utils.vertex_np_array_from_meshes(meshes)
            faces = meshes[0].faces

        return vertices,vertices_for_mean,faces

    def get_custom_basis_from_dataset(self,path,k=None,cache="./cached_meshes/"):
        vertices,vertices_for_mean,faces = self.load_dataset(path,cache=cache)
        self.faces = torch.from_numpy(np.array(faces))
        self.n_vertices = vertices.shape[1]
        scale,basis,mean_shape = get_pca_basis_from_np_vertices(vertices,vertices_for_mean,self.device,k=k,unit_norm=False)

        mean_shape = mean_shape.view(vertices[0].shape).cpu().detach().numpy()
        basis = basis.view(vertices[0].shape[0],vertices[0].shape[1],-1).cpu().detach().numpy()
        return scale,basis,mean_shape

    def load_dataset_round(self,path,round,cache="./cached_meshes"):
        if os.path.isdir(path) :
            registered_meshes,names = utils.read_objs_from_dir(path)
            vertices = utils.vertex_np_array_from_meshes(registered_meshes)
            faces = registered_meshes[0].faces
            vertices_for_mean = np.array([vertices[i] for i in range(len(vertices)) if "Wrapped" in names[i]])
            if round == 0:
                vertices = vertices_for_mean
            else:
                selected_names = [names[i] for i in range(len(vertices)) if (names[i].split(".obj")[0][-1]=="T" or int(names[i].split(".obj")[0].split("_r")[-1]) <= round )]
                vertices = np.array([vertices[i] for i in range(len(vertices)) if names[i] in selected_names])
            if not os.path.exists(cache):
                os.makedirs(cache)
            utils.dump_array_as_pickle(registered_meshes,os.path.join(cache,"meshes.pkl"))
        else:
            meshes = utils.load_array_from_pickle(path)
            vertices = utils.vertex_np_array_from_meshes(meshes)
            faces = meshes[0].faces

        return vertices,vertices_for_mean,faces

    def get_round_basis_from_dataset(self,path,round,k=None,cache="./cached_meshes/"):
        vertices,vertices_for_mean,faces = self.load_dataset_round(path,round,cache=cache)
        self.faces = torch.from_numpy(np.array(faces))
        self.n_vertices = vertices.shape[1]
        #scale,basis,mean_shape = get_pca_basis_from_np_vertices(vertices,vertices_for_mean,self.device,k=k,unit_norm=False)
        scale,basis,mean_shape = get_pca_basis_from_np_vertices(vertices,vertices_for_mean,self.device,k=self.num_betas,unit_norm=False)

        mean_shape = mean_shape.view(vertices[0].shape).cpu().detach().numpy()
        basis = basis.view(vertices[0].shape[0],vertices[0].shape[1],-1).cpu().detach().numpy()
        return scale,basis,mean_shape

    def forward(self, pose, betas , trans, v_shaped=None):
        '''
            STAR forward pass given pose, betas (shape) and trans
            return the model vertices and transformed joints
        :param pose: pose  parameters - A batch size x 72 tensor (3 numbers for each joint)
        :param beta: beta  parameters - A batch size x number of betas
        :param beta: trans parameters - A batch size x 3
        :return:
                 v         : batch size x 6890 x 3
                             The STAR model vertices
                 v.v_vposed: batch size x 6890 x 3 model
                             STAR vertices in T-pose after adding the shape
                             blend shapes and pose blend shapes
                 v.v_shaped: batch size x 6890 x 3
                             STAR vertices in T-pose after adding the shape
                             blend shapes and pose blend shapes
                 v.J_transformed:batch size x 24 x 3
                                Posed model joints.
                 v.f: A numpy array of the model face.
        '''
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs  = self.shapedirs.view(-1, self.num_betas)[None, :].expand(batch_size, -1, -1)
        beta = betas[:, :, None]

        if v_shaped is None:
            v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template

        J = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor])

        pose_quat = quat_feat(pose.view(-1, 3)).view(batch_size, -1)
        pose_feat = torch.cat((pose_quat[:,4:],beta[:,1]),1)

        R = rodrigues(pose.view(-1, 3)).view(batch_size, 24, 3, 3)
        R = R.view(batch_size, 24, 3, 3)

        posedirs = self.posedirs[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, pose_feat[:, :, None]).view(-1, 6890, 3)
        
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights, G.permute(1, 0, 2, 3).contiguous().view(24, -1)).view(6890, batch_size, 4,4).transpose(0, 1)
        rest_shape_h = torch.cat([v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        v = v + trans[:,None,:]
        v.f = self.f
        v.v_posed = v_posed
        v.v_shaped = v_shaped

        root_transform = with_zeros(torch.cat((R[:,0],J[:,0][:,:,None]),2)) #,self.device)
        results =  [root_transform]
        for i in range(0, self.parent.shape[0]):
            transform_i = with_zeros(torch.cat((R[:, i + 1], J[:, i + 1][:,:,None] - J[:, self.parent[i]][:,:,None]), 2)) #,self.device)
            curr_res = torch.matmul(results[self.parent[i]],transform_i)
            results.append(curr_res)
        results = torch.stack(results, dim=1)
        posed_joints = results[:, :, :3, 3]
        v.J_transformed = posed_joints + trans[:,None,:]
        mse = MSELoss()
        return v

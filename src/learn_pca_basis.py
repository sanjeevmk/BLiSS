import sys
from unicodedata import decomposition
import utils
import argparse
import os
import torch
import trimesh
from torch import nn
import numpy as np
#from star.pytorch.star import STAR

def get_pca_basis_from_np_vertices(np_vertices,np_vertices_for_mean,device,k=None,unit_norm=False):
    torch_vertices = torch.from_numpy(np_vertices).float().to(device)
    torch_vertices_mean = torch.from_numpy(np_vertices_for_mean).float().to(device)
    batch,n_vertices,n_dims = torch_vertices.size()
    batch_mean,n_vertices,n_dims = torch_vertices_mean.size()
    matrix = torch_vertices.view(batch,-1)
    matrix_mean = torch_vertices_mean.view(batch_mean,-1)
    mean = torch.mean(matrix_mean,0)
    matrix_norm = (matrix-mean.unsqueeze(0)) #/std.unsqueeze(0)
    U,_,V = torch.svd(matrix_norm)

    '''
    basis_matrix = V
    single_data = matrix[0]
    mean_coeffs = (single_data-mean).unsqueeze(0).matmul(basis_matrix)
    reconstruction = mean_coeffs[:,:40].matmul(basis_matrix.transpose(0,1)[:40,:]) + mean
    mse_loss = nn.MSELoss()
    error = mse_loss(reconstruction,single_data.unsqueeze(0))
    print(mean.size(),basis_matrix.size())
    print(mean_coeffs.size())
    print(error)
    exit()
    '''

    deviation = torch.std(matrix_norm @ V,0)
    if not unit_norm:
        for i in range(deviation.size()[0]):
            V[:,i] *= deviation[i]

    if k is not None:
        deviation = deviation[:k]
        V = V[:,:k]
    V = V.contiguous()
    return deviation,V,mean

def star_pca_basis(k):
    star = STAR(gender='neutral',num_betas=k)
    V = star.shapedirs.view(-1,star.shapedirs.size()[-1])
    mean =  star.v_template.view(1,-1).squeeze()
    S = torch.linalg.norm(V,dim=0)
    return S,V,mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('registered_dir',type=str)
    parser.add_argument('-pickle',action='store_true')
    parser.add_argument('-cache',type=str)
    parser.add_argument('-k',type=int)
    parser.add_argument('-output',type=str)


    args = parser.parse_args()

    if args.pickle:
        registered_meshes,_ = utils.read_objs_from_dir(args.registered_dir)

    if args.pickle and args.cache:
        if not os.path.exists(args.cache):
            os.makedirs(args.cache)

        registered_vertices = utils.vertex_np_array_from_meshes(registered_meshes)

        utils.dump_array_as_pickle(registered_vertices,os.path.join(args.cache,'registered.pkl'))
        utils.dump_array_as_pickle(registered_meshes,os.path.join(args.cache,'registered_meshes.pkl'))

    if args.cache and not args.pickle:
        registered_vertices = utils.load_array_from_pickle(os.path.join(args.cache,'registered.pkl'))
        registered_meshes = utils.load_array_from_pickle(os.path.join(args.cache,'registered_meshes.pkl'))

    deviation,basis_matrix,mean = get_pca_basis_from_np_vertices(registered_vertices,k=args.k)
    #deviation,basis_matrix,mean = star_pca_basis(args.k)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.cache:
        utils.dump_array_as_pickle(deviation,os.path.join(args.cache,'scale.pkl'))
        utils.dump_array_as_pickle(basis_matrix,os.path.join(args.cache,'basis.pkl'))
        utils.dump_array_as_pickle(mean.unsqueeze(0),os.path.join(args.cache,'mean.pkl'))

    mean_shape = mean.view(len(registered_meshes[0].vertices),3)
    _mesh = trimesh.Trimesh(vertices=mean_shape.cpu().detach().numpy(),faces=registered_meshes[0].faces,process=False)
    _mesh.export(os.path.join(args.output,'mean_shape.obj'))

    mean_coeffs = (mean-mean).unsqueeze(0).matmul(basis_matrix)
    reconstruction = mean_coeffs.matmul(basis_matrix.transpose(0,1)) + mean
    mse_loss = nn.MSELoss()
    error = mse_loss(reconstruction,mean.unsqueeze(0))
    '''
    for n_components in range(basis_matrix.size()[1]):
        reconstruction = mean_coeffs[:,:n_components].matmul(basis_matrix.transpose(0,1)[:n_components,:])
        vertices = reconstruction.view(len(registered_meshes[0].vertices),3).cpu().detach().numpy()
        error = mse_loss(reconstruction,mean.unsqueeze(0))
        _mesh = trimesh.Trimesh(vertices=vertices,faces=registered_meshes[0].faces,process=False)
        if not os.path.exists(os.path.join(args.output,'mean_recon_upto_k')):
            os.makedirs(os.path.join(args.output,'mean_recon_upto_k'))

        _mesh.export(os.path.join(args.output,'mean_recon_upto_k',str(n_components)+'.obj'))
    '''

    for n_components in range(basis_matrix.size()[1]):
        print(n_components)
        idx=0
        start = -1 
        end = 1
        interval = (end-start)/10.0
        for t in np.arange(start,end+interval,interval):
            reconstruction = t*basis_matrix.transpose(0,1)[n_components,:] + mean
            vertices = reconstruction.view(len(registered_meshes[0].vertices),3).cpu().detach().numpy()
            _mesh = trimesh.Trimesh(vertices=vertices,faces=registered_meshes[0].faces,process=False)
            if not os.path.exists(os.path.join(args.output,'varying_only_k',str(n_components))):
                os.makedirs(os.path.join(args.output,'varying_only_k',str(n_components)))

            _mesh.export(os.path.join(args.output,'varying_only_k',str(n_components),str(idx)+'.obj'))
            idx+=1

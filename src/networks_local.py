import torch
torch.manual_seed(4)
torch.cuda.manual_seed_all(4)
import torch.nn as nn
import torch.nn.functional as F
#from pytorch3d.ops import GraphConv
import numpy as np
from torch.nn.functional import max_pool1d

class CentroidNormalWKSFeat(nn.Module):
    def __init__(self,triangle_feature_dim):
        super(CentroidNormalWKSFeat,self).__init__()
        self.fc1 = nn.Linear(6+100,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,triangle_feature_dim)
        self.gn2 = nn.GroupNorm(4,128)

    def forward(self,centroids,normals,wks):
        x = torch.cat([centroids,normals,wks],-1)
        #x = wks
        xt = F.relu(self.fc1(x))
        xt = F.relu(self.fc2(xt))
        feature = self.fc3(xt)
        #feature = F.relu(self.fc3(xt2))

        return feature

class ScanGlobalFeature(nn.Module):
    def __init__(self,scan_feature_dim):
        super(ScanGlobalFeature,self).__init__()
        self.fc1 = nn.Linear(3,256)
        self.fc2 = nn.Linear(256,scan_feature_dim)

    def forward(self,points):
        xt = F.relu(self.fc1(points.squeeze()))
        feature = self.fc2(xt) #.permute(0,2,1)
        feature = torch.cat([points.squeeze(),feature],-1)
        feature_max = max_pool1d(feature,feature.size()[-1])

        return feature


class AE(nn.Module):
    def __init__(self,scan_feature_dim):
        super(AE,self).__init__()
        self.fc1 = nn.Linear(3,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,scan_feature_dim)

        self.dfc1 = nn.Linear(scan_feature_dim,512)
        self.dfc2 = nn.Linear(512,512)
        self.dfc3 = nn.Linear(512,6890*3)


    def encoder(self,points):
        #points = points.view(points.size()[0],-1)
        xt = F.relu(self.fc1(points))
        xt = F.relu(self.fc2(xt))
        feature = self.fc3(xt)
        feature,_ = torch.max(feature,dim=1)
        return feature

    def decoder(self,code):
        x = F.relu(self.dfc1(code))
        x = F.relu(self.dfc2(x))
        dec_points  = self.dfc3(x).view(code.size()[0],-1,3)
        return dec_points

    def forward(self,points):
        code = self.encoder(points)
        dec_points = self.decoder(code)
        return code,dec_points

class JacobianNetwork(nn.Module):
    def __init__(self,triangle_feature_dim,scan_feature_dim,local_feature_dim):
        super(JacobianNetwork,self).__init__()
        #self.fc1 = nn.Linear(2*triangle_feature_dim+scan_feature_dim+9,128)
        #self.fc2 = nn.Linear(128,128)
        #self.fc3 = nn.Linear(128,9)
        self.fc1 = nn.Conv1d(triangle_feature_dim+2*scan_feature_dim+local_feature_dim+3+9,128,1)
        #self.fc1 = nn.Conv1d(6+3+9,128,1)
        #self.fc1 = nn.Conv1d(triangle_feature_dim+2*scan_feature_dim+9,128,1)
        self.fc2 = nn.Conv1d(128,128,1)
        self.fc3 = nn.Conv1d(128,128,1)
        self.fc4 = nn.Conv1d(128,128,1)
        self.fc5 = nn.Conv1d(128,9,1)
        #self.fc4 = nn.Linear(128,9)
        self.gn1 = nn.GroupNorm(4,128)
        self.gn2 = nn.GroupNorm(4,128)
        self.gn3 = nn.GroupNorm(4,128)
        self.gn4 = nn.GroupNorm(4,128)
        #self.learnable_global = torch.nn.Parameter(torch.randn(13776,128))

    #def forward(self,triangle_features,target_points,inp_jacobians):
    def forward(self,triangle_features,input_global_feature,target_global_features,target_local_features,target_points,inp_jacobians):
        features = torch.cat([triangle_features,input_global_feature,target_global_features,target_local_features,target_points,inp_jacobians],-1)
        #features = torch.cat([triangle_features,target_points,inp_jacobians],-1)
        features = features.permute(1,0).unsqueeze(0)
        xt = F.relu(self.fc1(features))
        xt = F.relu(self.fc2(xt))
        xt = F.relu(self.fc3(xt))
        xt = F.relu(self.fc4(xt))
        djacobians = self.fc5(xt).permute(0,2,1).squeeze()
        #out_jacobians = jacobians.view(1,jacobians.size()[0],3,3)
        out_jacobians = inp_jacobians + djacobians
        out_jacobians = out_jacobians.view(1,out_jacobians.size()[0],3,3)
        return out_jacobians

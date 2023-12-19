from curses import killchar
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops.knn import knn_gather, knn_points
import utils

import resnet
from pointnet_utils import PointNetEncoder
import kaolin as kal
import kaolin.io.obj as kobj 
import kaolin.render as krender 
import kaolin.io.utils as kutils
from kaolin.render.camera import generate_perspective_projection
# render & camera
from kaolin.render.camera import perspective_camera
from kaolin.ops.mesh import index_vertices_by_faces, face_normals
from kaolin.render.mesh import dibr_rasterization as dibr_rasterization_kaolin
import random 


class Encoder(nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024, im_size=64):
        super(Encoder, self).__init__()
        dim_hidden = [dim1, dim1*2, dim1*4, dim2, dim2]

        self.conv1 = nn.Conv2d(dim_in, dim_hidden[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=2, padding=2)

        self.bn1 = nn.BatchNorm2d(dim_hidden[0])
        self.bn2 = nn.BatchNorm2d(dim_hidden[1])
        self.bn3 = nn.BatchNorm2d(dim_hidden[2])

        self.fc1 = nn.Linear(dim_hidden[2]*math.ceil(im_size/8)**2, dim_hidden[3])
        self.fc2 = nn.Linear(dim_hidden[3], dim_hidden[4])
        self.fc3 = nn.Linear(dim_hidden[4], dim_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        return x


class Decoder(nn.Module):
    def __init__(self,filename_obj,sphere_num=128, dim_in=512, centroid_scale=0.1, bias_scale=1.0, max_radii=0.2,dp_ratio=1.0):
        super(Decoder, self).__init__()
        # load .obj
        
        self.template_mesh = utils.load_obj_spherenum('./data/obj/cylinder/circle.obj',sphere_num)
        self.spheres_mesh = utils.load_obj_spherenum('./data/obj/sphere/sphere_42.obj',sphere_num)
        self.cylinder_mesh = utils.load_obj_spherenum('./data/obj/cylinder/square2.obj',sphere_num*2)

        self.register_buffer('vertices_base', self.template_mesh.vertices.cpu())  # vertices_base)
        self.register_buffer('faces', self.template_mesh.faces.cpu())  # faces)
        self.register_buffer('sphere_vertices_base', self.spheres_mesh.vertices.cpu())  # vertices_base)
        self.register_buffer('sphere_faces', self.spheres_mesh.faces.cpu())  # faces)
        self.sphere_num = sphere_num
        self.nv = self.vertices_base.size(0)
        self.nf = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5
        self.max_radii = max_radii
        self.dp_ratio = dp_ratio

        dim = 1024
        dim_hidden = [1024, 2048]
        self.fc1 = nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        #self.fc3 = nn.Linear(dim_hidden[1], dim_hidden[1])
        self.fc_radii = nn.Linear(dim_hidden[1], self.sphere_num * 1)
        self.fc_centroids= nn.Linear(dim_hidden[1],  self.sphere_num *3)

    def forward(self, x, viewpoint):
        # viewpoint 2B 3
        batch_size = x.shape[0]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        bias = self.fc_centroids(x) #* self.centroid_scale # B N 3
        bias = bias.clamp(min=-0.732,max=0.732)
        radii = torch.sigmoid(self.fc_radii(x)) * self.max_radii # B N 1  从0.1改为0.3

        radii = radii.clamp(min=0.02,max=self.max_radii)
        
 
        bias_init =  bias.view(-1, self.sphere_num , 3)
        bias = bias.view(-1, self.sphere_num , 3) #  B N 3
   
        radii_init = radii.view(-1, self.sphere_num , 1)
        radii = radii.view(-1, self.sphere_num , 1) # B N 1

        bias_double = torch.cat([bias,bias],dim=0) # 2B N 3
        radii_double = torch.cat([radii,radii],dim=0) # 2B N 3
        viewpoint = viewpoint.unsqueeze(1)

        v = F.normalize(-1*(bias_double[:,:,0:3] - viewpoint),dim=1) # 2B N 3
        edge_vector = v
        P = torch.Tensor([10.0,10.0,10.0]).unsqueeze(0).unsqueeze(0).cuda()
        P1 = (viewpoint - P).repeat([1,self.sphere_num,1])
        v1 = F.normalize(torch.cross(edge_vector,P1,dim=2),dim=2)
        v2 = F.normalize(torch.cross(edge_vector,v1,dim=2),dim=2)
        radii_repeat = radii_double.repeat([1,1,3])

        p1 = bias_double
        p2 = bias_double + radii_repeat * v1
        p3 = bias_double + radii_repeat * F.normalize((v1 + v2),dim=2)
        p4 = bias_double + radii_repeat * v2
        p5 = bias_double + radii_repeat * F.normalize((v2 - v1),dim=2)
        p6 = bias_double - radii_repeat * v1
        p7 = bias_double - radii_repeat * F.normalize((v1 + v2),dim=2)
        p8 = bias_double - radii_repeat * v2
        p9 = bias_double - radii_repeat * F.normalize((v2 - v1),dim=2)

        vertices = torch.stack([p1,p2,p3,p4,p5,p6,p7,p8,p9],dim=2)
        new_vertices = torch.reshape(vertices,(batch_size*2,-1,3))

        faces_all = self.faces[None, :, :].repeat(batch_size*2, 1, 1).cuda()

        if self.dp_ratio < 1.0:
            real_num = int( self.sphere_num * self.dp_ratio)
            real_face_num = int(self.faces.shape[0] / self.sphere_num * real_num)

            select_spheres, select_vertices = self.drop_out(vertices,torch.cat([bias,radii],dim=-1),self.dp_ratio)
        
 
            select_vertices = torch.reshape(select_vertices,(batch_size*2,-1,3))
            faces_drop = self.faces[None, :, :].repeat(batch_size*2, 1, 1).cuda()[:,0:real_face_num,:]

            return new_vertices,faces_all,bias,radii,select_spheres,select_vertices,faces_drop
        return new_vertices,faces_all,bias,radii,None,new_vertices,faces_all

    def transfer_circle(self,spheres,viewpoint):
        batch_size = spheres.shape[0]
        bias = spheres[:,:,0:3]
        radii = spheres[:,:,3]
        bias_double = torch.cat([bias,bias],dim=0) # 2B N 3
        radii_double = torch.cat([radii,radii],dim=0).unsqueeze(-1) # 2B N 3

        viewpoint = viewpoint.unsqueeze(1)

        v = F.normalize(-1*(bias_double[:,:,0:3] - viewpoint),dim=1) # 2B N 3
        edge_vector = v
        P = torch.Tensor([10.0,10.0,10.0]).unsqueeze(0).unsqueeze(0).cuda()
        P1 = (viewpoint - P).repeat([1,self.sphere_num*2,1])
 
        v1 = F.normalize(torch.cross(edge_vector,P1,dim=2),dim=2)
        v2 = F.normalize(torch.cross(edge_vector,v1,dim=2),dim=2)
        radii_repeat = radii_double.repeat([1,1,3])

        p1 = bias_double

        p2 = bias_double + radii_repeat * v1
        p3 = bias_double + radii_repeat * F.normalize((v1 + v2),dim=2)
        p4 = bias_double + radii_repeat * v2
        p5 = bias_double + radii_repeat * F.normalize((v2 - v1),dim=2)
        p6 = bias_double - radii_repeat * v1
        p7 = bias_double - radii_repeat * F.normalize((v1 + v2),dim=2)
        p8 = bias_double - radii_repeat * v2
        p9 = bias_double - radii_repeat * F.normalize((v2 - v1),dim=2)

        vertices = torch.stack([p1,p2,p3,p4,p5,p6,p7,p8,p9],dim=2)
        new_vertices = torch.reshape(vertices,(batch_size*2,-1,3))
        #print(self.faces.shape)
        #exit()
        faces_all = self.faces[None, :, :].repeat(batch_size*2, 1, 1).cuda()
        return new_vertices,faces_all
        
    def drop_out(self,vertices,spheres,p=1.0):
        '''
        选择比例为p的球，生成mesh。
        spheres： B N 4
        '''
        if p == 1.0:
            return spheres
        
        B,N,C = spheres.shape
        P = int(N*p)
        index = torch.LongTensor(random.sample(range(N),P)).cuda()

        select = torch.index_select(spheres,1,index)
        select_v = torch.index_select(vertices,1,index) 
        return select,select_v
        


class Decoder_DIS(nn.Module):
    def __init__(self,sphere_num=128):
        super(Decoder_DIS, self).__init__()
        # load .obj
        self.point_encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=4)
        self.sphere_num = sphere_num
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.sphere_num * 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        # x: B N 4
        batch_size = x.shape[0]
        x = x.transpose(2,1)


        global_feat, trans, trans_feat = self.point_encoder(x) # B 1024

        x = F.relu(self.fc1(global_feat), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        displacements = torch.sigmoid(self.fc3(x)) #* -1 # B N 1

        return displacements


class Model(nn.Module):
    def __init__(self,fine_num,surf_num, filename_obj, max_radii,dp_ratio,args):
        super(Model, self).__init__()

        # auto-encoder

        self.encoder_fine = resnet.resnet18(pretrained=True,num_classes=1000)
        # sphere_num
        
        self.decoder_fine = Decoder(filename_obj,sphere_num=fine_num,dim_in=1000,dp_ratio=dp_ratio)
       
        self.encoder_surf = resnet.resnet18(pretrained=True,num_classes=1000)
        # sphere_num

        self.decoder_surf = Decoder(filename_obj,sphere_num=surf_num,dim_in=1000,dp_ratio=dp_ratio)
        self.decoder_displacements = Decoder_DIS(sphere_num=fine_num+surf_num)

        
        
        self.cylinder_mesh = utils.load_obj_spherenum('./data/obj/cylinder/cube.obj',(fine_num+surf_num)*2)
        
    def model_param(self):
        return list(self.encoder_fine.parameters()) + list(self.decoder_fine.parameters()) + list(self.encoder_surf.parameters()) + list(self.decoder_surf.parameters()) + list(self.decoder_displacements.parameters())
    def model_param_finetune(self):
        return list(self.decoder_displacements.parameters())

    def reconstruct(self, images, viewpoint):
        vertices_fine, faces_fine,bias_fine, radii_fine,select_s_fine,select_v_fine,select_f_fine = self.decoder_fine(self.encoder_fine(images),viewpoint)
        vertices_surf, faces_surf,bias_surf, radii_surf,select_s_surf,select_v_surf,select_f_surf = self.decoder_surf(self.encoder_surf(images),viewpoint)

        if select_s_fine == None:
            select_s = None
        else:
            select_s = torch.cat([select_s_fine,select_s_surf],dim=1)

        select_v = torch.cat([select_v_fine,select_v_surf],dim=1)

        select_f = torch.cat([select_f_fine,select_f_fine+select_v_fine.shape[1]],dim=1)

        return vertices_fine, faces_fine,bias_fine, radii_fine,vertices_surf, faces_surf,bias_surf, radii_surf,select_s,select_v,select_f

    def reconstruct_surf(self, images, viewpoint):
        vertices_surf, faces_surf,bias_surf, radii_surf = self.decoder_surf(self.encoder_surf(images),viewpoint)
        return vertices_surf, faces_surf,bias_surf, radii_surf

    def reconstruct_semantic(self, images, viewpoint, semantic):
        if semantic == 'FINE':
            vertices, faces,bias, radii,select_s,select_v,select_f = self.decoder_fine(self.encoder_fine(images),viewpoint)
        else:
            vertices, faces,bias, radii,select_s,select_v,select_f = self.decoder_surf(self.encoder_surf(images),viewpoint)
        return vertices, faces,bias, radii,select_s,select_v,select_f
    
    def r_finetune(self,bias,radii,k=2):
        spheres = torch.cat([bias,radii],dim=-1)

        displacements = self.decoder_displacements(spheres)
        
        radii_finetune = (radii * displacements.unsqueeze(-1)).clamp(min=0.01)

        nearest_spheres,nearest_idx = self.get_neighbors(bias,radii_finetune,k) # B N 4*K

        vertices,faces = self.compute_edges_vf_1(bias,radii_finetune,nearest_spheres,k)

        return vertices,faces,radii_finetune,nearest_idx

    def compute_edges_vf_1(self,centers,radii,nearest_spheres,k):
        '''
        bias : B N 3
        radii: B N 1
        nearest_spheres: B N K 4
        '''

        sphere_num = centers.shape[1]
        batch_size = centers.shape[0]
        top_spheres = torch.cat([centers,radii],dim=-1) # B N 4

        top_spheres = top_spheres.unsqueeze(2).repeat(1,1,k,1)

        bottom_spheres = nearest_spheres # B N K 4

        
        top_spheres_edge = torch.reshape(top_spheres,(batch_size,sphere_num*k,4))
  
        bottom_spheres_edge = torch.reshape(bottom_spheres,(batch_size,sphere_num*k,4))

        vertices,faces = self.transfer_2squares(top_spheres_edge,bottom_spheres_edge)
        return vertices,faces



    def get_nearest_edges(self,centers,radii):
        ''' 
        centers B 128 3
        radii B 128 1
        '''
        length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
        length = length.cuda()
        nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=2)
        nearest_centers = knn_gather(centers, nn.idx[:,:,1].unsqueeze(-1), None)
        nearest_radii = knn_gather(radii, nn.idx[:,:,1].unsqueeze(-1), None)
        nearest_spheres = torch.cat([nearest_centers,nearest_radii],dim=3)
        return nearest_spheres[:,:,0,:].squeeze()

    def render_multiview(self, image_a, image_b, viewpoint_a, viewpoint_b):
        # [Ia, Ib]
        outputs = {}
        batch_size = image_a.shape[0]
        images = torch.cat((image_a, image_b), dim=0)

        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)
        outputs['viewpoints'] = viewpoints

        vertices_f, faces_f,bias_f, radii_f,vertices_s, faces_s,bias_s, radii_s = self.reconstruct(images,viewpoints)
        
        
        bias_all = torch.cat(([bias_f,bias_s]),dim=1)
        radii_all = torch.cat(([radii_f,radii_s]),dim=1)
        vertices = torch.cat(([vertices_f,vertices_s]),dim=1)
        faces = torch.cat(([faces_f,faces_s+vertices_f.shape[1]]),dim=1)
        


        outputs['spheres'] = torch.cat([bias_all,radii_all],dim=2)

        # [Ma, Mb, Ma, Mb]
        outputs['vertices'] = vertices
        outputs['faces'] = faces


        return outputs

    def render_multiview_w_fine(self, image_a, image_b, viewpoint_a, viewpoint_b):
        # [Ia, Ib]
        outputs = {}
        batch_size = image_a.shape[0]
        images = torch.cat((image_a, image_b), dim=0)


        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)
        outputs['viewpoints'] = viewpoints

        vertices_f, faces_f,bias_f, radii_f,vertices_s, faces_s,bias_s, radii_s, select_s,select_v,select_f = self.reconstruct(images,viewpoints)
        
        
        bias_all = torch.cat(([bias_f,bias_s]),dim=1)
        radii_all = torch.cat(([radii_f,radii_s]),dim=1)
        vertices = torch.cat(([vertices_f,vertices_s]),dim=1)
        faces = torch.cat(([faces_f,faces_s+vertices_f.shape[1]]),dim=1)
        
        outputs['centers'] = bias_all
        outputs['radii'] = radii_all
        outputs['centers_f'] = bias_f
        outputs['radii_f'] = radii_f
        outputs['centers_s'] = bias_s
        outputs['radii_s'] = radii_s

        outputs['spheres'] = torch.cat([bias_all,radii_all],dim=2)
        outputs['spheres_image_a'] = outputs['spheres'][0:batch_size,...]
        outputs['spheres_image_b'] = outputs['spheres'][batch_size:,...]

        outputs['vertices'] = vertices
        outputs['faces'] = faces

        v_num = vertices.shape[1]
        f_num = faces.shape[1]

        outputs['vertices_drop'] = select_v
        outputs['faces_drop'] = select_f 
        outputs['vertices_fine'] = vertices[:,:v_num//2,:]
        outputs['faces_fine'] = faces[:,:f_num//2,:] 

        return outputs


    def pred_spheres(self,image_a, image_b, viewpoint_a, viewpoint_b):
        outputs = {}
        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)
        outputs['viewpoints'] = viewpoints
        batch_size = image_a.shape[0]
        images = torch.cat((image_a, image_b), dim=0)

        vertices_f, faces_f,bias_f, radii_f,vertices_s, faces_s,bias_s, radii_s, select_s,select_v,select_f = self.reconstruct(images,viewpoints)
        
        
        bias_all = torch.cat(([bias_f,bias_s]),dim=1)
        radii_all = torch.cat(([radii_f,radii_s]),dim=1)

        
        outputs['centers'] = bias_all
        
        outputs['radii'] = radii_all


        return outputs


    def render_multiview_semantic(self, image_a, image_b, viewpoint_a, viewpoint_b,semantic):
        outputs = {}
        batch_size = image_a.shape[0]
        images = torch.cat((image_a, image_b), dim=0)
        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)
        outputs['viewpoints'] = viewpoints

        vertices, faces,bias, radii,select_s,select_v,select_f = self.reconstruct_semantic(images,viewpoints,semantic)
 
        outputs['centers'] = bias
        outputs['radii'] = radii
        outputs['select_spheres'] = select_s

        outputs['spheres'] = torch.cat([bias,radii],dim=2)
        outputs['spheres_image_a'] = outputs['spheres'][0:batch_size,...]
        outputs['spheres_image_b'] = outputs['spheres'][batch_size:,...]

        outputs['vertices'] = vertices
        outputs['faces'] = faces
        outputs['vertices_drop'] = select_v
        outputs['faces_drop'] = select_f

        return outputs




    def render_multiview_split(self, image_a, viewpoint_a):
        outputs = {}
        batch_size = image_a.shape[0]
        images = image_a

        viewpoints = viewpoint_a

        
        vertices_f, faces_f,bias_f, radii_f = self.reconstruct_semantic(images,None,'FINE')
        vertices_s, faces_s,bias_s, radii_s = self.reconstruct_semantic(images,None,'SURF')

        outputs['vertices_f'] = vertices_f
        outputs['faces_f'] = faces_f
        outputs['vertices_s'] = vertices_s
        outputs['faces_s'] = faces_s
        return outputs


    def render_neighbor(self, image_a, image_b, viewpoint_a, viewpoint_b,neighbor_num):
        # [Ia, Ib]
        k = neighbor_num
        outputs = {}
        batch_size = image_a.shape[0]
        images = torch.cat((image_a, image_b), dim=0)

        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)
       
        vertices_f, faces_f,bias_f, radii_f,vertices_s, faces_s,bias_s, radii_s = self.reconstruct(images)

        bias_all = torch.cat(([bias_f,bias_s]),dim=1)
        radii_all = torch.cat(([radii_f,radii_s]),dim=1)
        vertices = torch.cat(([vertices_f,vertices_s]),dim=1)
        faces = torch.cat(([faces_f,faces_s+vertices_f.shape[1]]),dim=1)
        f_v,f_faces,nearest_idx = self.edge_prediction(bias_all,radii_all,k)

        faces = torch.cat(([faces,f_faces+vertices.shape[1]]),dim=1)
        vertices = torch.cat(([vertices,f_v]),dim=1)

        outputs['centers'] = bias_all
        outputs['radii'] = radii_all
        outputs['spheres'] = torch.cat([bias_all,radii_all],dim=2)
        outputs['spheres_image_a'] = outputs['spheres'][0:batch_size,...]
        outputs['spheres_image_b'] = outputs['spheres'][batch_size:,...]
        outputs['nearest_idx'] = nearest_idx
        # [Ma, Mb, Ma, Mb]
        vertices = torch.cat((vertices, vertices), dim=0)
        faces = torch.cat((faces, faces), dim=0)

        return outputs

    def edge_prediction(self,bias,radii,k):
        nearest_spheres,nearest_idx = self.get_neighbors(bias,radii,k) # B N 4*K

        vertices,faces = self.compute_edges_vf(bias,radii,nearest_spheres,k)

        return vertices,faces,nearest_idx
    def get_neighbors(self,centers,radii,k):

        length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
        length = length.cuda()
        nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=k+1)

        nearest_centers = knn_gather(centers, nn.idx[:,:,1:], None).squeeze(2) # B N K 3

        nearest_radii = knn_gather(radii, nn.idx[:,:,1:], None).squeeze(2) # B N K 1
        return torch.cat([nearest_centers,nearest_radii],dim=-1), nn.idx[:,:,1:] # B N K 4
    def compute_edges_vf(self,centers,radii,nearest_spheres,k):
        '''
        bias : B N 3
        radii: B N 1
        nearest_spheres: B N K 4
        '''
        sphere_num = centers.shape[1]
        batch_size = centers.shape[0]
        top_spheres = torch.cat([centers,radii],dim=-1) # B N 4
        top_spheres = top_spheres.unsqueeze(2).repeat(1,1,k,1)

        bottom_spheres = nearest_spheres # B N K 4


        top_spheres_edge = torch.reshape(top_spheres,(batch_size,sphere_num*k,4))

        bottom_spheres_edge = torch.reshape(bottom_spheres,(batch_size,sphere_num*k,4))

        vertices,faces = self.transfer_2squares(top_spheres_edge,bottom_spheres_edge)
        return vertices,faces

    def render_multiview_progressive(self, image_a, image_b, viewpoint_a, viewpoint_b,current_epoch,max_epoch):
        # [Ia, Ib]
        outputs = {}
        batch_size = image_a.shape[0]
        images = torch.cat((image_a, image_b), dim=0)

        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)

        vertices_f, faces_f,bias_f, radii_f,vertices_s, faces_s,bias_s, radii_s = self.reconstruct(images)
        
        
        bias_all = torch.cat(([bias_f,bias_s]),dim=1)
        radii_all = torch.cat(([radii_f,radii_s]),dim=1)
        vertices = torch.cat(([vertices_f,vertices_s]),dim=1)
        faces = torch.cat(([faces_f,faces_s+vertices_f.shape[1]]),dim=1)
        
        outputs['centers'] = bias_all
        outputs['radii'] = radii_all
        outputs['centers_f'] = bias_f
        outputs['radii_f'] = radii_f
        outputs['centers_s'] = bias_s
        outputs['radii_s'] = radii_s

        outputs['spheres'] = torch.cat([bias_all,radii_all],dim=2)
        outputs['spheres_image_a'] = outputs['spheres'][0:batch_size,...]
        outputs['spheres_image_b'] = outputs['spheres'][batch_size:,...]

        # [Ma, Mb, Ma, Mb]
        vertices = torch.cat((vertices, vertices), dim=0)
        faces = torch.cat((faces, faces), dim=0)

        return outputs
    
    def render_multiview_split_fine(self, image_a, viewpoint_a):
        outputs = {}
        batch_size = image_a.shape[0]
        images = image_a

        viewpoints = viewpoint_a


        vertices_f, faces_f,bias_f, radii_f,_,_,_ = self.reconstruct_semantic(images,viewpoints,'FINE')

        outputs['vertices_f'] = vertices_f
        outputs['faces_f'] = faces_f
        outputs['centers'] = bias_f
        outputs['radii'] = radii_f

        outputs['spheres'] = torch.cat([bias_f,radii_f],dim=2)

        return outputs

    def render_multiview_edges(self, image_a, image_b, viewpoint_a, viewpoint_b,aimuth_a, aimuth_b):
        outputs = {}
        batch_size = image_a.shape[0]
        images = torch.cat((image_a, image_b), dim=0)

        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)

        sphere_vertices, sphere_faces,bias,radii = self.reconstruct(images)
        
                # [Ma, Mb, Ma, Mb]
        sphere_vertices = torch.cat((sphere_vertices, sphere_vertices), dim=0)
        sphere_faces = torch.cat((sphere_faces, sphere_faces), dim=0)

       
        outputs['centers'] = bias
        outputs['radii'] = radii
        outputs['spheres'] = torch.cat([bias,radii],dim=2)

        outputs['spheres_image_a'] = outputs['spheres'][0:batch_size,...]
        outputs['spheres_image_b'] = outputs['spheres'][batch_size:,...]

        bottom_spheres = self.get_nearest_edges(bias,radii)

        top_spheres = torch.cat([bias,radii],dim=2)

        new_vertices, faces = self.transfer_2squares(top_spheres,bottom_spheres)



        # [Ma, Mb, Ma, Mb]
        vertices = torch.cat((new_vertices, new_vertices), dim=0)
        faces = torch.cat((faces, faces), dim=0)

        return outputs

    def render_multiview_r_finetune(self, image_a, image_b, viewpoint_a, viewpoint_b, centers, radii):
        outputs = {}
        batch_size = image_a.shape[0]


        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)
        outputs['viewpoints'] = viewpoints

        edge_vertices,edge_faces,radii_finetune,nearest_idx= self.r_finetune(centers, radii)

        outputs['radii_finetune'] = radii_finetune

        vertices = torch.cat((edge_vertices, edge_vertices), dim=0)
        faces = torch.cat((edge_faces, edge_faces), dim=0)
        outputs['vertices'] = vertices
        outputs['faces'] = faces
        outputs['centers'] = centers
        outputs['radii'] = radii
        outputs['spheres'] = torch.cat([centers,radii],dim=2)
   
        return outputs


    def evaluate(self, images,viewpoints):
        '''
        输出：
        centers  radii radii_finetune
        render finetune 球
        '''
        outputs = {}

        vertices_f, faces_f,bias_f, radii_f,vertices_s, faces_s,bias_s, radii_s,_,_,_ = self.reconstruct(images,viewpoints)
        bias_all = torch.cat(([bias_f,bias_s]),dim=1)
        radii_all = torch.cat(([radii_f,radii_s]),dim=1)
        vertices = torch.cat(([vertices_f,vertices_s]),dim=1)
        faces = torch.cat(([faces_f,faces_s+vertices_f.shape[1]]),dim=1)


        edge_vertices,edge_faces,radii_finetune,nearest_idx= self.r_finetune(bias_all, radii_all)

        outputs['centers'] = bias_all
        outputs['radii_all'] = radii_all
        outputs['radii_finetune'] = radii_finetune

        outputs['spheres'] = torch.cat([bias_all,radii_all],dim=2)
        outputs['spheres_finetune'] = torch.cat([bias_all,radii_finetune],dim=2)
        
        return outputs

    def evaluate_overview(self, images,viewpoints):
        '''
        输出：
        centers  radii radii_finetune
        render finetune 球
        '''
        outputs = {}
        outputs['viewpoints'] = viewpoints
        vertices_f, faces_f,bias_f, radii_f,vertices_s, faces_s,bias_s, radii_s,_,_,_ = self.reconstruct(images,viewpoints)
        bias_all = torch.cat(([bias_f,bias_s]),dim=1)
        radii_all = torch.cat(([radii_f,radii_s]),dim=1)
        vertices = torch.cat(([vertices_f,vertices_s]),dim=1)
        faces = torch.cat(([faces_f,faces_s+vertices_f.shape[1]]),dim=1)
        
        edge_vertices,edge_faces,radii_finetune,nearest_idx= self.r_finetune(bias_all, radii_all)


        outputs['vertices_f'] = vertices_f
        outputs['faces_f'] = faces_f
        outputs['vertices_s'] = vertices_s
        outputs['faces_s'] = faces_s
        outputs['vertices'] = vertices
        outputs['faces'] = faces

        outputs['centers'] = bias_all
        outputs['radii_all'] = radii_all
        outputs['radii_finetune'] = radii_finetune

        outputs['spheres'] = torch.cat([bias_all,radii_all],dim=2)
        outputs['spheres_finetune'] = torch.cat([bias_all,radii_finetune],dim=2)
        outputs['vertices_finetune'],_ = self.decoder_fine.transfer_circle(outputs['spheres_finetune'],viewpoints)
        return outputs

    def evaluate_split(self, images,viewpoints):
        '''
        输出：
        centers  radii radii_finetune
        render finetune 球
        '''
        outputs = {}
        outputs['viewpoints'] = viewpoints
        vertices_f, faces_f,bias_f, radii_f,vertices_s, faces_s,bias_s, radii_s,_,_,_ = self.reconstruct(images,viewpoints)
        bias_all = torch.cat(([bias_f,bias_s]),dim=1)
        radii_all = torch.cat(([radii_f,radii_s]),dim=1)
        vertices = torch.cat(([vertices_f,vertices_s]),dim=1)
        faces = torch.cat(([faces_f,faces_s+vertices_f.shape[1]]),dim=1)

        outputs['vertices_f'] = vertices_f
        outputs['faces_f'] = faces_f
        outputs['vertices_s'] = vertices_s
        outputs['faces_s'] = faces_s
        outputs['vertices'] = vertices
        outputs['faces'] = faces

        outputs['centers'] = bias_all
        outputs['radii'] = radii_all

        outputs['spheres'] = torch.cat([bias_all,radii_all],dim=2)


        return outputs

    def evaluate_render_edge(self, images,viewpoints):
        '''
        输出：
        centers  radii radii_finetune
        render finetune 球
        '''
        k=2
        outputs = {}
        batch_size = images.shape[0]

        vertices_f, faces_f,bias_f, radii_f,vertices_s, faces_s,bias_s, radii_s = self.reconstruct(images,torch.cat([viewpoints,viewpoints],dim=0))
        bias_all = torch.cat(([bias_f,bias_s]),dim=1)[0:batch_size]
        radii_all = torch.cat(([radii_f,radii_s]),dim=1)[0:batch_size]
        vertices = torch.cat(([vertices_f,vertices_s]),dim=1)[0:batch_size]
        faces = torch.cat(([faces_f,faces_s+vertices_f.shape[1]]),dim=1)[0:batch_size]
        outputs['vertices'] = vertices
        outputs['faces'] = faces

       


        edge_vertices,edge_faces,radii_finetune,nearest_idx= self.r_finetune(bias_all, radii_all,k)
        
        nearest_spheres,nearest_idx = self.get_neighbors(bias_all,radii_all,k) # B N 4*K
        vertices_edge_init,faces_edge_init = self.compute_edges_vf_1(bias_all,radii_all,nearest_spheres,k)
        outputs['centers'] = bias_all
        outputs['radii_all'] = radii_all
        outputs['radii_finetune'] = radii_finetune

        outputs['spheres'] = torch.cat([bias_all,radii_all],dim=2)
        outputs['spheres_finetune'] = torch.cat([bias_all,radii_finetune],dim=2)
        
        vertices = edge_vertices
        faces = edge_faces
        outputs['vertices_edge'] = edge_vertices
        outputs['faces_edge'] = edge_faces

        return outputs


    def evaluate_joint(self, images,viewpoints):
        '''
        输出：
        centers  radii radii_finetune
        render finetune 球
        '''
        outputs = {}
        batch_size = images.shape[0]

        outputs['viewpoints'] = viewpoints

        vertices_f, faces_f,bias_f, radii_f,vertices_s, faces_s,bias_s, radii_s, select_s,select_v,select_f = self.reconstruct(images,viewpoints)
        
        bias_all = torch.cat(([bias_f,bias_s]),dim=1)
        radii_all = torch.cat(([radii_f,radii_s]),dim=1)
        vertices = torch.cat(([vertices_f,vertices_s]),dim=1)
        faces = torch.cat(([faces_f,faces_s+vertices_f.shape[1]]),dim=1)


        outputs['spheres'] = torch.cat([bias_all,radii_all],dim=2)


        outputs['vertices'] = vertices
        outputs['faces'] = faces


        return outputs

    def evaluate_surf(self, image_a,viewpoint_a):
        '''
        
        '''
        outputs = {}
        batch_size = image_a.shape[0]
        images = image_a

        viewpoints = viewpoint_a

        vertices_s, faces_s,bias_s, radii_s,_,_,_ = self.reconstruct_semantic(images,viewpoints,'SURF')

        outputs['vertices_s'] = vertices_s
        outputs['faces_s'] = faces_s
        outputs['bias_s'] = bias_s
        outputs['radii_s'] = radii_s

        outputs['spheres'] = torch.cat([bias_s,radii_s],dim=2)
        
        return outputs


    def transfer_2squares(self,top_spheres,bottom_spheres):
        data_list = {}
        batch_size = top_spheres.shape[0]
        faces = self.cylinder_mesh.faces.repeat(2,1,1).cuda() #  E*F 3


        edge_num = top_spheres.shape[1] # here use sphere_num
        dis = torch.sqrt(torch.sum(torch.pow(top_spheres[:,:,0:3] - bottom_spheres[:,:,0:3],2),2))
        mask = torch.le(dis,0.1).unsqueeze(-1)  # B N 1

        top_centers = top_spheres[:,:,0:3]
        top_radii = top_spheres[:,:,3].unsqueeze(-1) * 1.414 * mask
        bottom_centers = bottom_spheres[:,:,0:3]
        bottom_radii = bottom_spheres[:,:,3].unsqueeze(-1) * 1.414 * mask
        edge_vector = top_centers - bottom_centers 

        P = torch.Tensor([10.0,10.0,10.0]).unsqueeze(0).unsqueeze(0).cuda()

        P1 = top_centers - P
        v1 = torch.abs(F.normalize(torch.cross(edge_vector,P1,dim=2),dim=2))
        v2 = F.normalize(torch.cross(edge_vector,v1,dim=2),dim=2)

        p0 = top_centers + top_radii.repeat([1,1,3]) * v1
        p1 = top_centers + top_radii.repeat([1,1,3]) * v2
        p2 = top_centers - top_radii.repeat([1,1,3]) * v1
        p3 = top_centers - top_radii.repeat([1,1,3]) * v2
        p4 = bottom_centers + bottom_radii.repeat([1,1,3]) * v1
        p5 = bottom_centers + bottom_radii.repeat([1,1,3]) * v2
        p6 = bottom_centers - bottom_radii.repeat([1,1,3]) * v1
        p7 = bottom_centers - bottom_radii.repeat([1,1,3]) * v2
        vertices = torch.stack([p0,p1,p2,p3,p4,p5,p6,p7],dim=2)


        new_vertices = torch.reshape(vertices,(batch_size,-1,3))
        return new_vertices,faces

    def evaluate_iou(self, images, voxels):
        vertices, faces,bias,radii = self.reconstruct(images)

        faces_ = srf.face_vertices(vertices, faces).data
        faces_norm = faces_ * 1. * (32. - 1) / 32. + 0.5
        voxels_predict = srf.voxelization(faces_norm, 32, False).cpu().numpy()
        voxels_predict = voxels_predict.transpose(0, 2, 1, 3)[:, :, :, ::-1]
        iou = (voxels * voxels_predict).sum((1, 2, 3)) / (0 < (voxels + voxels_predict)).sum((1, 2, 3))
        bias = bias.cpu().detach().numpy()
        radii = radii.cpu().detach().numpy()
        return iou, vertices, faces, bias,radii


    def evaluate_edge(self, images, viewpoints):
        vertices, faces,bias,radii = self.reconstruct(images)
        
        bottom_spheres = self.get_nearest_edges(bias,radii)
        top_spheres = torch.cat([bias,radii],dim=2)
        new_vertices, faces = self.transfer_2squares(top_spheres,bottom_spheres)
       
        bias = bias.cpu().detach().numpy()
        radii = radii.cpu().detach().numpy()
        return new_vertices, faces, bias,radii

    def predict_spheres(self,images):
        outputs = {}
        batch_size = images.shape[0]
        
        vertices_f, faces_f,bias_f, radii_f,vertices_s, faces_s,bias_s, radii_s = self.reconstruct(images,None)
        bias_all = torch.cat(([bias_f,bias_s]),dim=1)
        radii_all = torch.cat(([radii_f,radii_s]),dim=1)

        vertices = torch.cat(([vertices_f,vertices_s]),dim=1)
        faces = torch.cat(([faces_f,faces_s+vertices_f.shape[1]]),dim=1)
        outputs['vertices'] = vertices
        outputs['faces'] = faces
        outputs['centers'] = bias_all
        outputs['radii'] = radii_all
        outputs['spheres'] = torch.cat([bias_all,radii_all],dim=2)

        return outputs

    def predict_spheres_render(self,images,viewpoints):
        outputs = {}
        batch_size = images.shape[0]

        vertices_f, faces_f,bias_f, radii_f,vertices_s, faces_s,bias_s, radii_s = self.reconstruct(images,None)
        bias_all = torch.cat(([bias_f,bias_s]),dim=1)
        radii_all = torch.cat(([radii_f,radii_s]),dim=1)
        vertices = torch.cat(([vertices_f,vertices_s]),dim=1)
        faces = torch.cat(([faces_f,faces_s+vertices_f.shape[1]]),dim=1)
        outputs['vertices'] = vertices
        outputs['faces'] = faces
        outputs['centers'] = bias_all
        outputs['radii'] = radii_all
        outputs['spheres'] = torch.cat([bias_all,radii_all],dim=2)
        return outputs


    def forward(self, images=None, viewpoints=None, azimuths=None, voxels=None,centers=None, radii=None,task='train',current_epoch=None,max_epoch=None,neighbor_num=1):
        if task == 'train':
            return self.render_multiview(images[0], images[1], viewpoints[0], viewpoints[1])
        elif task == 'train_w_fine':
            return self.render_multiview_w_fine(images[0], images[1], viewpoints[0], viewpoints[1])
        elif task == 'train_surf':
            return  self.render_multiview_semantic(images[0], images[1], viewpoints[0], viewpoints[1],'SURF')
        elif task == 'train_fine':
            return  self.render_multiview_semantic(images[0], images[1], viewpoints[0], viewpoints[1],'FINE')
        elif task == 'finetune_r':
            return  self.render_multiview_r_finetune(images[0], images[1], viewpoints[0], viewpoints[1], centers , radii)
        elif task == 'pred_spheres':
            return self.pred_spheres(images[0], images[1], viewpoints[0], viewpoints[1])
        elif task == 'test_surf':
            return self.evaluate_surf(images[0],viewpoints[0])
        elif task == 'test':
            return self.evaluate(images[0], viewpoints[0])
        elif task == 'test_overview':
            return self.evaluate_overview(images[0], viewpoints[0])
        elif task == 'test_split':
            return self.evaluate_split(images[0], viewpoints[0])
        elif task == 'test_render_edge':
            return self.evaluate_render_edge(images, viewpoints)
        elif task == 'test_fine':
            return self.render_multiview_split_fine(images[0],viewpoints[0])
        elif task == 'test_split':
            return self.render_multiview_split(images[0],viewpoints[0])
        elif task == 'test_joint':
            return self.evaluate_joint(images[0], viewpoints[0])
        elif task == 'test_edge':
            return self.evaluate_edge(images,viewpoints)
        elif task == 'train_edges':
            return self.render_multiview_edges(images[0], images[1], viewpoints[0], viewpoints[1],azimuths[0],azimuths[1])
        elif task == 'train_progressive':
            return self.render_multiview_progressive(images[0], images[1], viewpoints[0], viewpoints[1],current_epoch,max_epoch)
        elif task == 'predict_spheres':
            return self.predict_spheres(images[0], images[1], viewpoints[0], viewpoints[1])

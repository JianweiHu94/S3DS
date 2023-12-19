import argparse

import torch
import torch.nn.parallel
import datasets_split_view0_dataloader_testsplit as datasets
from torch.utils.data import Dataset, DataLoader
from utils import AverageMeter, img_cvt, transfer_spheres

import models_merge_circle_resnet_1113 as models_merge
import time
import os
import cv2
import imageio
import numpy as np
import math
from collections import OrderedDict
from camera_util import rotate_vertices
from kaolin.render.camera import generate_perspective_projection
from utils_functions import render_vertex_colors
import utils


FINE_NUM=200
SURF_NUM=200
MAX_RADII=0.2
BATCH_SIZE = 24

IMAGE_SIZE = 64
CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')


PRINT_FREQ = 100
SAVE_FREQ = 100
DP_RATIO = 1.0

MODEL_DIRECTORY = 'data/results/models'
DATASET_DIRECTORY = 'data/shapenet_img/test/viewangle_30_224_split_test'

SIGMA_VAL = 1e-6
IMAGE_PATH = ''


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-eid', '--experiment-id', type=str)
parser.add_argument('-md', '--model-directory', type=str, default=MODEL_DIRECTORY)
parser.add_argument('-dd', '--dataset-directory', type=str, default=DATASET_DIRECTORY)
parser.add_argument('-cls', '--class-ids', type=str, default='')
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
parser.add_argument('-bs', '--batch-size', type=int, default=BATCH_SIZE)
parser.add_argument('-img', '--image-path', type=str, default=IMAGE_PATH)

parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)

parser.add_argument('-pf', '--print-freq', type=int, default=PRINT_FREQ)
parser.add_argument('-sf', '--save-freq', type=int, default=SAVE_FREQ)
#parser.add_argument('--shading-model', action='store_true', help='test shading model')
args = parser.parse_args()


model_joint = models_merge.Model(FINE_NUM,SURF_NUM,'data/obj/sphere/sphere_42.obj',MAX_RADII,DP_RATIO, args=args)
model_joint = model_joint.cuda()
CLASS_IDS_CHAIR = (args.class_ids)
state_dicts = torch.load(args.model_directory)
model_joint.load_state_dict(state_dicts['model'], strict=True)
model_joint.eval()

dataset = datasets.ShapeNet(CLASS_IDS_CHAIR.split(',')[0], 'test')
dataset_train = DataLoader(dataset,batch_size=BATCH_SIZE,num_workers=10,shuffle=False)
directory_output = './data/results/test'
os.makedirs(directory_output, exist_ok=True)
directory_mesh = os.path.join(directory_output, args.experiment_id)
os.makedirs(directory_mesh, exist_ok=True)


camproj_mtx = generate_perspective_projection(fovyangle=30 / 180.0 * np.pi,
                                                  ratio=1.0).cuda()

def test():
    end = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()

    iou_all = []

    for class_id in CLASS_IDS_CHAIR.split(','):

        directory_mesh_cls = os.path.join(directory_mesh, class_id)
        os.makedirs(directory_mesh_cls, exist_ok=True)
        os.makedirs(directory_mesh_cls + '/png/', exist_ok=True)
        os.makedirs(directory_mesh_cls + '/prob_idx/', exist_ok=True)
        os.makedirs(directory_mesh_cls + '/xyzr/', exist_ok=True)
        os.makedirs(directory_mesh_cls + '/xyzr_init/', exist_ok=True)
        os.makedirs(directory_mesh_cls + '/xyz/', exist_ok=True)
        os.makedirs(directory_mesh_cls + '/obj/', exist_ok=True)
        os.makedirs(directory_mesh_cls + '/voxel/', exist_ok=True)
        iou = 0.0
        for i,batch_data in enumerate(dataset_train):
            images_a, images_b, viewpoints_a, viewpoints_b, mask_a,mask_b,resnet_a,resnet_b,s_m_a,s_m_b,s_m_a_init,s_m_b_init = numpy2cuda(batch_data) 
            #print(resnet_a)
            #exit()
            surf_mask_a = (mask_a - s_m_a).clamp(min=0)
            surf_mask_b = (mask_b - s_m_b).clamp(min=0)

            outputs  = model_joint([resnet_a],[viewpoints_a.repeat(2,1)],task='test',)

            for k in range(images_a.shape[0]):
                # if k != 1:
                #     continue
                img = np.zeros((224,224,3),'int32')
                demo_mask = mask_a[k,0,...].cpu().detach().numpy()
                obj_id = (i * args.batch_size + k)

                mesh_path_init = os.path.join(directory_mesh_cls + '/obj/', '%04d_%02d_init.obj' %(i,k))
                mesh_path_finetune = os.path.join(directory_mesh_cls + '/obj/', '%04d_%02d_finetune.obj' %(i,k))
                mesh_path_edge = os.path.join(directory_mesh_cls + '/obj/', '%04d_%02d_edge.obj' %(i,k))
                mesh_path_fine = os.path.join(directory_mesh_cls + '/obj/', '%04d_%02d_fine.obj' %(i,k))
                mesh_path_surf = os.path.join(directory_mesh_cls + '/obj/', '%04d_%02d_surf.obj' %(i,k))
                input_path = os.path.join(directory_mesh_cls + '/png/', '%04d_%02d.png' %(i,k))
                render_path_init = os.path.join(directory_mesh_cls + '/png/', '%04d_%02d_render_init.png' %(i,k))
                render_path_finetune = os.path.join(directory_mesh_cls + '/png/', '%04d_%02d_render_finetune.png' %(i,k))
                render_path_edge = os.path.join(directory_mesh_cls + '/png/', '%04d_%02d_render_edge.png' %(i,k))
                mask_path = os.path.join(directory_mesh_cls + '/png/', '%04d_%02d_mask.png' % (i,k))
                sphere_path = os.path.join(directory_mesh_cls + '/xyzr/', '%04d_%02d.xyzr' %(i,k))
                sphere_init_path = os.path.join(directory_mesh_cls + '/xyzr_init/', '%04d_%02d_init.xyzr' %(i,k))
                center_path = os.path.join(directory_mesh_cls + '/xyz/', '%04d_%02d.xyz' %(i,k))
                spheres = outputs['spheres'][k]
                spheres_finetune = outputs['spheres_finetune'][k]
                fine_spheres = outputs['spheres'][k][0:FINE_NUM]
                surf_spheres = outputs['spheres'][k][FINE_NUM:]
                #utils.write_ma(sphere_init_path,spheres.cpu().detach().numpy())
                #utils.write_ma(sphere_path,spheres_finetune.cpu().detach().numpy())
                
                if k == 1:
                    utils.write_ma(center_path,spheres_finetune[:,0:3].cpu().detach().numpy())
                    vertices,faces = transfer_spheres(spheres_finetune)
                    utils.save_obj(mesh_path_finetune, vertices, faces)
                
                
                #srf.save_obj(mesh_path_init, vertices_init[k], faces_init[k])
                #srf.save_obj(mesh_path_init, vertices[k], faces[k])
                #srf.save_obj(mesh_path_fine, v_f[k], f_f[k])
                #srf.save_obj(mesh_path_surf, v_s[k], f_s[k])
                #imageio.imsave(os.path.join(directory_mesh_cls + '/png/', '%04d_%02d_fake_surf.png' % (i,k)), np.array(img_cvt(render_s[k][3,:,:].unsqueeze(0).repeat(4,1,1))*(0.0,0.0,0.0,1.0),'uint8'))
                #imageio.imsave(os.path.join(directory_mesh_cls + '/png/', '%04d_%02d_fake_fine.png' % (i,k)), np.array(img_cvt(render_f[k][3,:,:].unsqueeze(0).repeat(4,1,1))*(0.0,0.0,0.0,1.0),'uint8'))
                #imageio.imsave(os.path.join(directory_mesh_cls + '/png/', '%04d_%02d_gt_surf.png' % (i,k)), np.array(img_cvt(surf_mask_a[k,:,:,:])*(0.0,0.0,0.0,1.0),'uint8'))
                #imageio.imsave(os.path.join(directory_mesh_cls + '/png/', '%04d_%02d_gt_fine.png' % (i,k)), np.array(img_cvt(s_m_a[k,:,:,:])*(0.0,0.0,0.0,1.0),'uint8'))
                #print(mask_a.shape)
                #cv2.imwrite(os.path.join(directory_mesh_cls + '/png/', '%07d_input_mask.png' % i),demo_mask*255)
                #exit()
                #imageio.imsave(render_path_finetune, img_cvt(render_finetune[k][3,:,:].unsqueeze(0).repeat(4,1,1)))
                #imageio.imsave(os.path.join(directory_mesh_cls + '/png/', '%04d_%02d_input_mask.png' % (i,k)), np.array(img_cvt(1-images_a[k,:,:,3].unsqueeze(0)),'uint8'))
                '''
                srf.save_obj(mesh_path_edge, edge_vertices[k], edge_faces[k])
                imageio.imsave(input_path, img_cvt(images[k]))

                imageio.imsave(mask_path, img_cvt(images[k,3,:,:].unsqueeze(0)))
                imageio.imsave(render_path_init, img_cvt(render_init[k][3,:,:].unsqueeze(0).repeat(4,1,1)))
                imageio.imsave(render_path_finetune, img_cvt(render_finetune[k][3,:,:].unsqueeze(0).repeat(4,1,1)))
                imageio.imsave(render_path_edge, img_cvt(render_edge[k][3,:,:].unsqueeze(0).repeat(4,1,1)))
                '''
               #exit()

def numpy2cuda(batch_data):
    images_a,images_b,viewpoints_a,viewpoints_b,mask_a,mask_b,resnet_a,resnet_b,s_m_a_init,s_m_b_init = batch_data

    mask_a = mask_a.unsqueeze(1).repeat(1,4,1,1).cuda()
    #print(mask_a.dtype)
    #print(s_m_a_init)
    #exit()
    mask_b = mask_b.unsqueeze(1).repeat(1,4,1,1).cuda()
    s_m_a = s_m_a_init.unsqueeze(1).repeat(1,4,1,1).cuda()
    s_m_b = s_m_b_init.unsqueeze(1).repeat(1,4,1,1).cuda()

    resnet_a = resnet_a.cuda()
    resnet_b = resnet_b.cuda()

    return  images_a,images_b,viewpoints_a.cuda(),viewpoints_b.cuda(),mask_a,mask_b,resnet_a,resnet_b,s_m_a,s_m_b,s_m_a_init,s_m_b_init


test()

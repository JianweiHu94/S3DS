import os 
import sys 
import cv2
import torch
from camera_util import rotate_vertices
from utils_functions import render_vertex_colors
from utils import AverageMeter, img_cvt, transfer_spheres
import utils
from kaolin.render.camera import generate_perspective_projection
import numpy as np

camproj_mtx = generate_perspective_projection(fovyangle=30 / 180.0 * np.pi,
                                                  ratio=1.0).cuda()



def render_prepare(batchsize,num,dp_ratio):
    meshcolors_render = np.ones([batchsize*4,(num)*9,3],'float32')
    meshcolors_render = torch.from_numpy(meshcolors_render).cuda()
    meshcolors_render_drop = np.ones([batchsize*4,int(num*dp_ratio)*9,3],'float32')
    meshcolors_render_drop = torch.from_numpy(meshcolors_render_drop).cuda()
    return meshcolors_render,meshcolors_render_drop

def render_prepare_edge(batchsize,num,dp_ratio):
    meshcolors_render = np.ones([batchsize*4,(num)*2*8,3],'float32')
    meshcolors_render = torch.from_numpy(meshcolors_render).cuda()
    meshcolors_render_drop = np.ones([batchsize*4,int(num*dp_ratio)*2*8,3],'float32')
    meshcolors_render_drop = torch.from_numpy(meshcolors_render_drop).cuda()
    return meshcolors_render,meshcolors_render_drop



def remder_mask(outputs,meshcolors_render,meshcolors_render_drop,drop=False):
    # render all
    if drop:
        vertices_rot,_,_ = rotate_vertices(outputs['vertices_drop'],outputs['viewpoints'])
        
        pred_ims, pred_masks, meshnormals = \
            render_vertex_colors(vertices_rot, outputs['faces_drop'][0], meshcolors_render_drop, camproj_mtx, 224, 224)
        render_images = pred_masks
        return render_images
    else:
        vertices_rot,_,_ = rotate_vertices(outputs['vertices'],outputs['viewpoints'])

        pred_ims, pred_masks, meshnormals = \
            render_vertex_colors(vertices_rot, outputs['faces'][0], meshcolors_render, camproj_mtx, 224, 224)
        render_images_all = pred_masks
        return render_images_all


def render_demo(epoch,v_f,f_f,meshcolors_render,meshcolors_render_drop,eval_data,outputs,cfg,cat='fine'):
    demo_path = os.path.join(cfg['directory_output'], 'demo_%07d_%s.obj' % (epoch,cat))
    
    eval_img,eval_mask,eval_tensor,eval_viewpoints = eval_data

    # render all
    vertices_rot,_,_ = rotate_vertices(v_f,eval_viewpoints)
    pred_ims, pred_masks, meshnormals = \
        render_vertex_colors(vertices_rot, f_f[0], meshcolors_render, camproj_mtx, 224, 224)
    render_f = pred_masks
    demo_image = eval_img[0:1]
    demo_mask = eval_mask[0,0,...]
    vertices,faces = transfer_spheres(outputs['spheres'][0])
    utils.save_obj(demo_path, vertices, faces)
    cv2.imwrite(os.path.join(cfg['image_output'], '%07d_fake_mask_%s.png' % (epoch,cat)),render_f.detach().cpu().numpy()[0]*255)


def render_batch_demo(epoch,v_f,f_f,meshcolors_render,meshcolors_render_drop,mask_a,outputs,cfg,cat='fine'):
    demo_path = os.path.join(cfg['directory_output'], 'demo_%07d_%s.obj' % (epoch,cat))
    viewpoints = outputs['viewpoints']
    # render all
    vertices_rot,_,_ = rotate_vertices(v_f,viewpoints)
    pred_ims, pred_masks, meshnormals = \
        render_vertex_colors(vertices_rot, f_f[0], meshcolors_render, camproj_mtx, 224, 224)
    render_f = pred_masks
    demo_mask = mask_a[0,0,...]
    vertices,faces = transfer_spheres(outputs['spheres'][0])
    utils.save_obj(demo_path, vertices, faces)
    cv2.imwrite(os.path.join(cfg['image_output'], '%07d_fake_mask_%s.png' % (epoch,cat)),render_f.detach().cpu().numpy()[0]*255)

import argparse
from tracemalloc import start
import cv2
import torch
import numpy as np
from losses import *
from utils import AverageMeter, img_cvt, transfer_spheres
import datasets_split_view0_dataloader_random as datasets
from datasets_split_view0_dataloader import get_eval_img

from torch.utils.data import Dataset, DataLoader
import models_merge_circle_resnet_1113 as models_merge

import imageio

import time
import os
from torch.utils.tensorboard import SummaryWriter


import utils
from render import *
import prepare


RESUME_PATH = ''

parser = argparse.ArgumentParser()

parser.add_argument('-eid', '--experiment-id', type=str)
parser.add_argument('-config', type=str, help='Path to config file.')
parser.add_argument('-md', '--model-directory', type=str, default='')

args = parser.parse_args()

cfg = utils.load_config(args.config)


torch.backends.cudnn.deterministic = True
torch.manual_seed(1234569527)
torch.cuda.manual_seed_all(1234569527)
_ = torch.manual_seed(1234569527)
np.random.seed(1234569527)

cfg = prepare.prepare(cfg,args)
model = prepare.create_model(cfg,args)

os.system('cp %s %s'%(args.config,cfg['directory_output']))



dataset = datasets.ShapeNet(cfg['train']['CLASS_ID'].split(',')[0], 'train')
dataset_train = DataLoader(dataset,batch_size=cfg['train']['BATCH_SIZE'],num_workers=10,shuffle=True)

directory_output = cfg['directory_output']

losses_dicts = prepare.set_loss(cfg)


def train():
    writer = SummaryWriter(cfg['directory_output'])
    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    optimizer = torch.optim.Adam(model.model_param(), cfg['train']['LEARNING_RATE'])
    
    #eval_data = get_eval_img()
    #eval_img,eval_mask,eval_tensor,eval_viewpoints = eval_data
    # train fine 
    # print('traning fine')

    if not cfg['train']['FINETUNE']:
        counts = 0
        NUM_ITERATIONS_FINE = cfg['train']['NUM_ITERATIONS_FINE']
        rgb,rgb_drop = render_prepare(cfg['train']['BATCH_SIZE'],cfg['model']['FINE_NUM'],cfg['model']['DP_RATIO'])
        for i,batch_data in enumerate(dataset_train):
            
            batch_data_use = numpy2cuda(batch_data) 
            images_a, images_b, viewpoints_a, viewpoints_b, mask_a,mask_b,resnet_a,resnet_b,s_m_a,s_m_b,s_m_a_init,s_m_b_init = batch_data_use
            lr = adjust_learning_rate([optimizer], cfg['train']['LEARNING_RATE'], i, method=cfg['train']['LR_TYPE'],change_iter=NUM_ITERATIONS_FINE//2)
            outputs  = model([resnet_a, resnet_b],[viewpoints_a, viewpoints_b],task='train_fine')
            render_images = remder_mask(outputs,rgb,rgb_drop,drop=False)
            overall_loss,loss,near_r_loss,rep_loss,s_loss,_ = get_fine_loss(render_images,s_m_a, s_m_b, outputs,losses_dicts,cfg,writer,i)
            losses.update(overall_loss.data.item(), images_a.size(0))
            
            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()

            if i % cfg['train']['DEMO_FREQ'] == 0:
                render_batch_demo(i,outputs['vertices'],outputs['faces'],rgb,rgb_drop,mask_a,outputs,cfg,cat='fine')
            if i % cfg['train']['PRINT_FREQ'] == 0:
                print('overall_loss: %.4f,  mtv: %.4f, repulsion: %.4f, s_loss: %.4f, near_r_loss: %.4f\n'%(overall_loss,loss,rep_loss,s_loss,near_r_loss))
            counts+=1 
            if counts == NUM_ITERATIONS_FINE:
                break

    if not cfg['train']['FINETUNE']:
        model_path = os.path.join(directory_output, 'checkpoint_fine_%07d.pth.tar' % counts)
        torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, model_path)

    print('traning surf')
    
    # counts=0
    # print(NUM_ITERATIONS_SURF)

    if not cfg['train']['FINETUNE']:
        counts = 0
        NUM_ITERATIONS_SURF = cfg['train']['NUM_ITERATIONS_SURF']
        rgb,rgb_drop = render_prepare(cfg['train']['BATCH_SIZE'],cfg['model']['SURF_NUM'],cfg['model']['DP_RATIO'])
        for i,batch_data in enumerate(dataset_train):
            
            batch_data_use = numpy2cuda(batch_data) 
            images_a, images_b, viewpoints_a, viewpoints_b, mask_a,mask_b,resnet_a,resnet_b,s_m_a,s_m_b,s_m_a_init,s_m_b_init = batch_data_use
            lr = adjust_learning_rate([optimizer], cfg['train']['LEARNING_RATE'], i, method=cfg['train']['LR_TYPE'],change_iter=NUM_ITERATIONS_SURF//2)
            surf_mask_a = (mask_a - s_m_a).clamp(min=0)
            surf_mask_b = (mask_b - s_m_b).clamp(min=0)
            outputs  = model([resnet_a, resnet_b],[viewpoints_a, viewpoints_b],task='train_surf')
            render_images = remder_mask(outputs,rgb,rgb_drop,drop=False)
            overall_loss,loss,near_r_loss,rep_loss,s_loss,_ = get_surf_loss(render_images,surf_mask_a, surf_mask_b, outputs,losses_dicts,cfg,writer,i)
            losses.update(overall_loss.data.item(), images_a.size(0))
           
            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()


            if i % cfg['train']['DEMO_FREQ'] == 0:
                #outputs  = model([eval_tensor.repeat(2,1,1,1)],[eval_viewpoints.repeat(4,1)],task='test_surf')
                #render_demo(i,outputs['vertices_s'],outputs['faces_s'],rgb,rgb_drop,eval_data,outputs,cfg,cat='surf')
                render_batch_demo(i,outputs['vertices'],outputs['faces'],rgb,rgb_drop,mask_a,outputs,cfg,cat='surf')
            if i % cfg['train']['PRINT_FREQ'] == 0:
                print('overall_loss: %.4f,  mtv: %.4f, repulsion: %.4f, s_loss: %.4f, near_r_loss: %.4f\n'%(overall_loss,loss,rep_loss,s_loss,near_r_loss))
            counts+=1 
            if counts == NUM_ITERATIONS_SURF:
                break

    if not cfg['train']['FINETUNE']:
        model_path = os.path.join(directory_output, 'checkpoint_surf_%07d.pth.tar' % counts)
        torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, model_path)

    print('traning joint')

    if not cfg['train']['FINETUNE']:
        counts=0
        NUM_ITERATIONS_JOINT = cfg['train']['NUM_ITERATIONS_JOINT']
        rgb,rgb_drop = render_prepare(cfg['train']['BATCH_SIZE'],cfg['model']['SURF_NUM']+cfg['model']['FINE_NUM'],cfg['model']['DP_RATIO'])

        for i,batch_data in enumerate(dataset_train):
            batch_data_use = numpy2cuda(batch_data) 
            images_a, images_b, viewpoints_a, viewpoints_b, mask_a,mask_b,resnet_a,resnet_b,s_m_a,s_m_b,s_m_a_init,s_m_b_init = batch_data_use
            lr = adjust_learning_rate([optimizer], cfg['train']['LEARNING_RATE'], i, method=cfg['train']['LR_TYPE'],change_iter=NUM_ITERATIONS_JOINT//2)
            outputs  = model([resnet_a, resnet_b],[viewpoints_a, viewpoints_b],task='train_w_fine',current_epoch=i, max_epoch=NUM_ITERATIONS_JOINT)
            render_images = remder_mask(outputs,rgb,rgb_drop,drop=False)
            overall_loss,loss,fine_loss,near_r_loss,rep_loss,s_loss,_ = get_joint_loss(render_images,mask_a, mask_b,s_m_a,s_m_b, outputs,losses_dicts,cfg,writer,i)
            losses.update(overall_loss.data.item(), images_a.size(0))
            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()
            if i % cfg['train']['DEMO_FREQ'] == 0:
                #outputs  = model([eval_tensor.repeat(2,1,1,1)],[eval_viewpoints.repeat(4,1)],task='test_joint')
                #render_demo(i,outputs['vertices'],outputs['faces'],rgb,rgb_drop,eval_data,outputs,cfg,cat='joint')
                render_batch_demo(i,outputs['vertices'],outputs['faces'],rgb,rgb_drop,mask_a,outputs,cfg,cat='joint')
            if i % cfg['train']['PRINT_FREQ'] == 0:
                print('overall_loss: %.4f,  mtv_joint: %.4f, repulsion: %.4f, s_loss: %.4f, near_r_loss: %.4f, fine_loss: %.4f\n'%(overall_loss,loss,rep_loss,s_loss,near_r_loss,fine_loss))
            counts+=1 
            if counts == NUM_ITERATIONS_JOINT:
                break


    if not cfg['train']['FINETUNE']:
        model_path = os.path.join(directory_output, 'checkpoint_joint_%07d.pth.tar' % counts)
        torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, model_path)


    # train r_finetune
    print('finetune')

    optimizer = torch.optim.Adam(model.model_param_finetune(), cfg['train']['LEARNING_RATE'])
    counts=0
    NUM_ITERATIONS_FINETUNE = cfg['train']['NUM_ITERATIONS_FINETUNE']
    rgb,rgb_drop = render_prepare_edge(cfg['train']['BATCH_SIZE'],cfg['model']['SURF_NUM']+cfg['model']['FINE_NUM'],1.0)

    for i, batch_data in enumerate(dataset_train):
        batch_data_use = numpy2cuda(batch_data) 
        images_a, images_b, viewpoints_a, viewpoints_b, mask_a,mask_b,resnet_a,resnet_b,s_m_a,s_m_b,s_m_a_init,s_m_b_init = batch_data_use
        lr = adjust_learning_rate([optimizer], cfg['train']['LEARNING_RATE'], i, method=cfg['train']['LR_TYPE'],change_iter=NUM_ITERATIONS_FINETUNE//2)
        outputs  = model([resnet_a, resnet_b],[viewpoints_a, viewpoints_b],task='pred_spheres')
        centers = outputs['centers'].detach()
        radii = outputs['radii'].detach()
        
        outputs  = model([resnet_a, resnet_b],[viewpoints_a, viewpoints_b],task='finetune_r',centers=centers,radii=radii)
        radii_finetune = outputs['radii_finetune']

        render_images = remder_mask(outputs,rgb,rgb_drop,drop=False)

        overall_loss,loss,fine_loss,near_r_loss,writer = get_finetune_loss(render_images,mask_a, mask_b, s_m_a, s_m_b, outputs,losses_dicts,cfg,writer,i)

        losses.update(overall_loss.data.item(), images_a.size(0))

        # compute gradient and optimize
        optimizer.zero_grad()
        overall_loss.backward()
        optimizer.step()

        if i % cfg['train']['DEMO_FREQ'] == 0:
            render_batch_demo(i,outputs['vertices'],outputs['faces'],rgb,rgb_drop,mask_a,outputs,cfg,cat='finetune')
        if i % cfg['train']['PRINT_FREQ'] == 0:
                print('overall_loss: %.4f,  mtv: %.4f, fine: %.4f, near_r_loss: %.4f\n'%(overall_loss,loss,fine_loss,near_r_loss))
        counts+=1 
        if counts == NUM_ITERATIONS_FINETUNE:
            break
    
    model_path = os.path.join(directory_output, 'checkpoint_finetune_%07d.pth.tar' % NUM_ITERATIONS_FINETUNE)
    torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, model_path)
    writer.flush()





def adjust_learning_rate(optimizers, learning_rate, i, method,change_iter=50000):
    if method == 'step':
        lr, decay = learning_rate, 0.3
        if i >= change_iter:
            lr *= decay
    elif method == 'constant':
        lr = learning_rate
    else:
        print("no such learing rate type")

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def set_lr(optimizers,learning_rate):
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

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


def adjust_sigma(sigma, i):
    # decay = 0.3
    # if i >= 50000:
    #     sigma *= decay
    return sigma

if __name__ == '__main__':
    train()
    

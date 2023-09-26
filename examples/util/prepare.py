import models_merge_circle_resnet_1113 as models_merge
from losses import *
import torch


def prepare(cfg,args):

    # mkdir folder
    directory_output = os.path.join('data/results/models', args.experiment_id)
    os.makedirs(directory_output, exist_ok=True)
    image_output = os.path.join(directory_output, 'pic')
    os.makedirs(image_output, exist_ok=True)
    cfg['directory_output'] = directory_output
    cfg['image_output'] = image_output
    return cfg

def create_model(cfg,args):
    FINE_NUM = cfg['model']['FINE_NUM']
    SURF_NUM = cfg['model']['SURF_NUM']
    MAX_RADII = cfg['model']['MAX_RADII']
    DP_RATIO = cfg['model']['DP_RATIO']
    model_joint = models_merge.Model(FINE_NUM,SURF_NUM,'data/obj/sphere/sphere_42.obj',MAX_RADII,dp_ratio=DP_RATIO,args=args)
    model_joint = model_joint.cuda()
    return model_joint


def set_loss():
    losses = {}
    knnRepul = kNNRepulsionLoss(k=10,n_seeds=20, h=0.01)
    kNN = kNNLoss(k=4,n_seeds=50)
    sphereLoss = SphereInterLoss(k=10)
    losses['knnRepul'] = knnRepul
    losses['kNN'] = kNN
    losses['sphereLoss'] = sphereLoss
    return losses

def render_prepare(batchsize,num,dp_ratio):
    meshcolors_render = np.ones([batchsize*4,(num)*9,3],'float32')
    meshcolors_render = torch.from_numpy(meshcolors_render).cuda()
    meshcolors_render_drop = np.ones([batchsize*4,int(num*dp_ratio)*9,3],'float32')
    meshcolors_render_drop = torch.from_numpy(meshcolors_render_drop).cuda()
    return meshcolors_render,meshcolors_render_drop
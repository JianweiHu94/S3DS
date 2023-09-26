import torch
import torch.nn as nn
import numpy as np
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops.knn import knn_gather, knn_points
from torchvision.transforms import Resize,ToPILImage,ToTensor
import torch.nn.functional as F
import math
import time

def iou(predict, target, eps=1e-6):
    dims = tuple(range(predict.ndimension())[1:])
    #print(dims)
    
    intersect = (predict * target).sum(dims)
    # print(intersect.shape)
    # exit()
    union = (predict + target - predict * target).sum(dims) + eps
    # (predict - predict * target).sum(dims) + eps
    # print(intersect.nelement())
    # if intersect.nelement() == 0:
    #     exit()
    return (intersect / union).sum() / (intersect.nelement())


def iou_loss(predict, target):
    return 1 - iou(predict, target)


def multiview_iou_loss(predicts, targets_a, targets_b):
    batch_size = targets_a.shape[0]
    loss = (iou_loss(predicts[0:batch_size,:,:,0], targets_a[:,3]) +
            iou_loss(predicts[batch_size:batch_size*2,:,:,0], targets_a[:,3]) +
            iou_loss(predicts[batch_size*2:batch_size*3,:,:,0], targets_b[:,3]) +
            iou_loss(predicts[batch_size*3:,:,:,0], targets_b[:,3])) / 4
    return loss

def multiview_iou_fine_loss(predicts, fine_a_mask, fine_b_mask, fine_a, fine_b):

    #相当于一个mask，只保留fine结构对应的预测，然后计算损失。
    #print(fine_a.shape)
    # loss = (iou_loss(predicts[0][:, 3] * fine_a_mask, fine_a) +
    #         iou_loss(predicts[1][:, 3] * fine_a_mask, fine_a) +
    #         iou_loss(predicts[2][:, 3] * fine_b_mask, fine_b) +
    #         iou_loss(predicts[3][:, 3] * fine_b_mask, fine_b)) / 4
    batch_size = fine_a_mask.shape[0]
    loss = (iou_loss(predicts[0:batch_size,:,:,0] * fine_a_mask, fine_a) +
            iou_loss(predicts[batch_size:batch_size*2,:,:,0] * fine_a_mask, fine_a) +
            iou_loss(predicts[batch_size*2:batch_size*3,:,:,0] * fine_b_mask, fine_b) +
            iou_loss(predicts[batch_size*3:,:,:,0] * fine_b_mask, fine_b)) / 4
    return loss

    return loss

def split_iou_loss(fine,surf, fine_a, surf_a, fine_b, surf_b):
    # print(fine[0].shape)
    # print(fine_a.shape)
    # print(torch.max(fine[0][0]))
    # print(torch.max(fine_a))
    # exit()
    loss = (iou_loss(fine[0][:, 3], fine_a) +
            iou_loss(surf[0][:, 3], surf_a) +
            iou_loss(fine[1][:, 3], fine_b) +
            iou_loss(surf[1][:, 3], surf_b)) / 4
    #print(loss.shape)
    return loss

def multiview_proj_iou_loss(predicts, targets_a, targets_b):
    proj_size = predicts[0].shape[1]
    gt_size = targets_a.shape[2]
    #targets_a = torch.unsqueeze(targets_a,dim=-1)
    #targets_b = torch.unsqueeze(targets_b,dim=-1)
    #print(targets_a.shape)
    #print(predicts[0].shape)
    #exit()
    #print(predicts[0][:,:,:,0].shape)
    #print(targets_a[:,:,:].shape)
    
    loss = (iou_loss(predicts[0][:,:,:,0], targets_a) +
            iou_loss(predicts[1][:,:,:,0], targets_a) +
            iou_loss(predicts[2][:,:,:,0], targets_b) +
            iou_loss(predicts[3][:,:,:,0], targets_b)) / 4.0
    #print(loss)
    #exit()
    return loss

def CD_loss(spheres_view1,spheres_view2,cd_weight,r_weight):
    batch_size = spheres_view1.shape[0]
    sphere_num = spheres_view1.shape[1]
    cd_loss,idx = chamfer_distance(spheres_view1[:,:,0:3],spheres_view2[:,:,0:3])
    #print(spheres_view1.shape)
    x_nn = knn_points(spheres_view1[:,:,0:3], spheres_view2[:,:,0:3], lengths1=None, lengths2=None, K=1)
    y_nn = knn_points(spheres_view2[:,:,0:3], spheres_view1[:,:,0:3], lengths1=None, lengths2=None, K=1)
    radii_view1 = spheres_view1[:,:,3].unsqueeze(-1)
    radii_view2 = spheres_view2[:,:,3].unsqueeze(-1)
    #print(x_nn.idx.shape)
    #print(radii_view1.shape)
    #print(x_nn.idx[0,:,0])
    R_loss = 0.0
    nearest_r2_2_r1 = knn_gather(radii_view2, x_nn.idx, None)[..., 0, :]
    #print(nearest_r2_2_r1.shape)
    nearest_r1_2_r2 = knn_gather(radii_view1, y_nn.idx, None)[..., 0, :]
    #print(nearest_r2_2_r1.shape)

    #print(t.shape)
    #t=torch.mean(t,dim=1)
    #t=torch.mean(t,dim=0)
    r_loss = torch.mean(torch.mean(torch.norm(radii_view1 - nearest_r2_2_r1,dim=2),dim=1),dim=0) \
            + torch.mean(torch.mean(torch.norm(radii_view2 - nearest_r1_2_r2,dim=2),dim=1),dim=0)
    #print(t.shape)
    return cd_weight*cd_loss,r_weight*r_loss
    

def regularize_bias(centers):
    # B N 3
    # 两个球之间的距离尽可能要大
    # 
    length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
    length = length.cuda()
    nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=2)
    #print(centers.shape)
    nearest_centers = knn_gather(centers, nn.idx[:,:,1].unsqueeze(-1), None).squeeze(2)
    #print(nearest_centers.shape)
    dis_reg = -1*torch.mean(torch.norm(centers - nearest_centers,dim=2))
    return dis_reg

def regularize_radii(radii):
    r_reg = torch.mean(radii)
    # B N 1 
    # 对平均半径进行约束，半径尽可能小。
    return r_reg

def edge_loss(centers,radii,k=1):
    pass
    # 假定每个球最近的一个球是有边，可以进行插值k个，从而得到N*K个新的球
    # 然后进行渲染

def nearest_smooth_loss(centers,radii,k=4):
    #pass
    # 对每个球选择K个邻近球，计算与K个球在坐标和半径上的平滑。
    length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
    length = length.cuda()
    nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=k+1)
    #print(centers.shape)
    #print(nn.idx[:,:,1:].shape)
    nearest_centers = knn_gather(centers, nn.idx[:,:,1:], None).squeeze(2)
    #print(nearest_centers.shape)
    nearest_radii = knn_gather(radii, nn.idx[:,:,1:], None).squeeze(2)
    #print(nearest_radii.shape)
    neighbor_center_mean = torch.mean(nearest_centers,dim=2)
    #print(neighbor_center_mean.shape)
    neighbor_radii_mean = torch.mean(nearest_radii,dim=2)
    #print(neighbor_radii_mean.shape)
    xyz_lap_loss = torch.mean(torch.norm(centers - neighbor_center_mean,dim=2))
    r_lap_loss = torch.mean(torch.norm(radii - neighbor_radii_mean,dim=2))
    #print(xyz_lap_loss)
    #print(r_lap_loss)
    lap_loss = xyz_lap_loss + r_lap_loss
    return xyz_lap_loss,r_lap_loss

def smooth_y_loss(centers,radii,k=16,threshold=0.02):
    #pass
    # 对每个球选择K个邻近球，计算与K个球在坐标和半径上的平滑。
    length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
    length = length.cuda()
    nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=k+1)

    nearest_centers = knn_gather(centers, nn.idx[:,:,1:], None).squeeze(2)

    nearest_radii = knn_gather(radii, nn.idx[:,:,1:], None).squeeze(2)
    neighbor_center_mean = torch.mean(nearest_centers,dim=2)
    neighbor_radii_mean = torch.mean(nearest_radii,dim=2)
    dis_y = torch.abs(neighbor_center_mean[:,:,1] - centers[:,:,1])
    #print(dis_y.shape)
    dis_r = torch.abs(neighbor_radii_mean- radii).squeeze()
    t = torch.lt(dis_y,0.04)
    #print(t.shape)
    #print(dis_r.shape)
    xyz_lap_loss = torch.mean(dis_y * t)
    r_lap_loss = torch.mean(dis_r * t)

    return xyz_lap_loss,r_lap_loss

def smooth_vector_loss(centers,radii,k=16,threshold=0.02):
    length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
    length = length.cuda()
    nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=k+1)

    nearest_centers = knn_gather(centers, nn.idx[:,:,1:], None).squeeze(2)

    nearest_radii = knn_gather(radii, nn.idx[:,:,1:], None).squeeze(2)
    neighbor_center_mean = torch.mean(nearest_centers,dim=2)
    neighbor_radii_mean = torch.mean(nearest_radii,dim=2)
    vector = neighbor_center_mean[:,:,0:3] - centers[:,:,0:3] # B N 3
    dis = torch.norm(vector,dim=2)
    dis_per_axis = torch.abs(vector)
    dis_sum = torch.sum(dis_per_axis,dim=2) # B N 
    t = torch.lt(dis,threshold)

    xyz_lap_loss = torch.mean(dis_sum * t)
    #r_lap_loss = torch.mean(dis_r * t)

    return xyz_lap_loss
    
def neighbors(centers,radii,k=4):
    #pass
    # 对每个球选择K个邻近球，计算与K个球在坐标和半径上的平滑。
    length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
    length = length.cuda()
    nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=k+1)
    #print(centers.shape)
    #print(nn.idx[:,:,1:].shape)
    nearest_centers = knn_gather(centers, nn.idx[:,:,1:], None).squeeze(2)
    #print(nearest_centers.shape)
    nearest_radii = knn_gather(radii, nn.idx[:,:,1:], None).squeeze(2)
    #print(nearest_radii.shape)
    neighbor_center_mean = torch.mean(nearest_centers,dim=2)
    #print(neighbor_center_mean.shape)
    neighbor_radii_mean = torch.mean(nearest_radii,dim=2)
    
    return neighbor_center_mean,neighbor_radii_mean

def smooth_loss_1(centers,radii,k=4):
    #pass
    # 对每个球选择K个邻近球，计算与K个球在坐标和半径上的平滑。
    length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
    length = length.cuda()
    nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=k+1)
    #print(centers.shape)
    #print(nn.idx[:,:,1:].shape)
    nearest_centers = knn_gather(centers, nn.idx[:,:,1:], None).squeeze(2)
    #print(nearest_centers.shape)
    nearest_radii = knn_gather(radii, nn.idx[:,:,1:], None).squeeze(2)
    #print(nearest_radii.shape)
    neighbor_center_mean = torch.mean(nearest_centers,dim=2)
    #print(neighbor_center_mean.shape)
    neighbor_radii_mean = torch.mean(nearest_radii,dim=2)
    #print(neighbor_radii_mean.shape)
    min_value,min_axis = torch.min(torch.abs(centers - neighbor_center_mean),dim=2)
    #print(min_value.shape)
    r_dis_loss = torch.mean(torch.abs(radii - neighbor_radii_mean))
    xyz_smooth_loss = torch.mean(min_value)
    loss = xyz_smooth_loss + r_dis_loss
    #print(r_dis_loss)
    #print(xyz_smooth_loss)
    return loss
    #r_lap_loss = torch.mean(torch.norm(radii - neighbor_radii_mean,dim=2))
    #print(min_value.shape)
    #print(min_axis[0,0:4])
    #print(r_lap_loss)
    #lap_loss = xyz_lap_loss + r_lap_loss
    #return lap_loss

def tangent_loss(centers,radii):
    #pass
    # 对每个球选择K个邻近球，计算与K个球在坐标和半径上的平滑。
    length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
    length = length.cuda()
    nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=2)

    nearest_centers = knn_gather(centers, nn.idx[:,:,1:], None).squeeze(2)
    print(nearest_centers.shape)
    exit()
    nearest_radii = knn_gather(radii, nn.idx[:,:,1:], None).squeeze(2)
    
    dis = torch.norm(centers - nearest_centers,dim=2)
    r_sum = radii + nearest_radii

    t_loss = torch.mean(torch.abs(dis - r_sum))
    return t_loss

def cluster_reg_loss(centers,radii):
    #pass
    # 对每个球选择K个邻近球，计算与K个球在坐标和半径上的平滑。
    length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
    length = length.cuda()
    nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=2)

    nearest_centers = knn_gather(centers, nn.idx[:,:,1:], None).squeeze(2)
    #print(nearest_centers.shape)
    nearest_radii = knn_gather(radii, nn.idx[:,:,1:], None).squeeze(2)
    dis = torch.norm(centers - nearest_centers,dim=2)
    mean_dis = torch.mean(dis,dim=1) # B 1
    norm_dim = torch.pow(dis - mean_dis,2)
    dis_loss = 1000*torch.mean(torch.mean(norm_dim,dim=1),dim=0)

    return dis_loss

def symmetry_loss(centers,radii):
    # chair 关于 yoz平面对称，也及时把x变负
    # centers B N 3
    centers_sym = centers * torch.Tensor([-1.0,1.0,1.0]).cuda()
    #centers_sym = centers
    #预先假设3D对称平面
    #cd_loss,idx = chamfer_distance(centers,centers_sym)
    #print(cd_loss)

    # exit()
    length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
    length = length.cuda()
    #print(spheres_view1.shape)
    x_nn = knn_points(centers, centers_sym, lengths1=length, lengths2=length, K=1)
    y_nn = knn_points(centers_sym, centers, lengths1=length, lengths2=length, K=1)
    #radii_view1 = spheres_view1[:,:,3].unsqueeze(-1)
    #radii_view2 = spheres_view2[:,:,3].unsqueeze(-1)
    nearest_centers_sym = knn_gather(centers_sym, x_nn.idx, None).squeeze(2)
    nearest_centers = knn_gather(centers, y_nn.idx, None).squeeze(2)
    #print(nearest_centers.shape)
    cd_loss = torch.mean(torch.mean(torch.norm(centers - nearest_centers_sym,dim=2),dim=1),dim=0) \
            + torch.mean(torch.mean(torch.norm(centers_sym - nearest_centers,dim=2),dim=1),dim=0)
    #print(cd_loss)
    #exit()
    #nearest_radii = knn_gather(radii, nn.idx[:,:,1:], None).squeeze(2)
    #R_loss = 0.0
    # nearest_radii_ = knn_gather(centers, x_nn.idx, None)
    # print(nearest_r2_2_r1.shape)
    # nearest_r1_2_r2 = knn_gather(centers_sym, y_nn.idx[:,:,1:], None)
    # print(nearest_r1_2_r2.shape)
    # exit()
    
    # #print(nearest_r2_2_r1.shape)

    # #print(t.shape)
    # #t=torch.mean(t,dim=1)
    # #t=torch.mean(t,dim=0)
    # r_loss = torch.mean(torch.mean(torch.norm(radii_view1 - nearest_r2_2_r1,dim=2),dim=1),dim=0) \
    #         + torch.mean(torch.mean(torch.norm(radii_view2 - nearest_r1_2_r2,dim=2),dim=1),dim=0)
    return cd_loss

def nearest_r_loss(centers,radii):
    length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
    length = length.cuda()
    nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=2)
    
    nearest_radii = knn_gather(radii, nn.idx[:,:,1:], None).squeeze(2)
    #print(nearest_radii.shape)
    loss = torch.mean(torch.abs(radii - nearest_radii))
    return loss


def repulsion_loss(centers,radii):
    length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
    length = length.cuda()
    nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=2)
    
    #nearest_radii = knn_gather(radii, nn.idx[:,:,1:], None).squeeze(2)
    nearest_centers= knn_gather(centers, nn.idx[:,:,1:], None).squeeze(2)
    #print(nearest_radii.shape)
    dis = torch.norm(centers - nearest_centers,dim=2)
    #print(dis)
    #print(dis.shape)
    loss = torch.mean(1.0/(dis**3+1e-6))
    return loss

def repulsion_loss_2(centers,radii):
    length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
    length = length.cuda()
    nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=2)
    
    #nearest_radii = knn_gather(radii, nn.idx[:,:,1:], None).squeeze(2)
    nearest_centers= knn_gather(centers, nn.idx[:,:,1:], None).squeeze(2)
    #print(nearest_radii.shape)
    dis = torch.norm(centers - nearest_centers,dim=2)
    #print(dis)
    #print(dis.shape)
    loss = torch.mean(1.0/(dis**3+1e-6))
    return loss

# 可以考虑重建出物体的三维属性。
# 如，不同视角的重建结果，应当是想象的，而不止图像的相似。
# 两个128个球的相似，可以用CD EMD，R loss来判定。
# 包括对称性

# 输入图像尺寸对于结果的影响，还是 GT mask的尺寸。
# 用更高尺寸的渲染结果。

# 对于AB视角生成的3D，都渲染在CD（随机）视角下，他们两个渲染出来的也应该
#是相似的。权重可以低一些。

# 如何把中轴球的性质体现出来。 如何解决平面上球过于大的问题
# 如果同时学习一个物体的表面网格，如何确保球是在内部？而且最大内切的性质。
# 对于一个面片，采样m个点，每个点都是可以有三角面片顶点来凸组合得到，并且记录。
# 然后对预测的球，找与球最近的点，内部的球 能不能定义一个符号函数

def egde_laplacian_loss(spheres,edge_matrix):
    '''
    edge_matrix B N N
    spheres B N 4

    '''
    x = spheres[:,:,0:4]
    batch_size = x.size(0)
    x = torch.matmul(edge_matrix, x)
    dims = tuple(range(x.ndimension())[1:])
    x = x.pow(2).sum(dims)
    return x.mean()

def edge_flatten_loss(spheres,edges,v_idx,v_mask,e_mask,condition=False):
    '''
    spheres B N 4
    edges B M 2  有填充
    v_idx  B M 24 每个边，所在面的另一顶点索引，有填充
    v_mask  B M 24 0或1， 可以表示v_idx当前值是否是有效的
    e_mask B M 1 表示边的掩码
    思路：
    计算边所有面的法向，或者高的向量。
    这样得到的是
    M 24 3的向量，有些是无效的，需要用v_mask标记

    '''
    eps = 1e-6
    threshold=math.cos(math.pi/4)
    batch_size = spheres.shape[0]
    edge_num = edges.shape[1]
    centers = spheres[:,:,0:3]
    idx0 = edges[:,:,0]
    idx1 = edges[:,:,1]
    v0 = knn_gather(centers, idx0.unsqueeze(-1), None).squeeze(2) # B M 3
    v1 = knn_gather(centers, idx1.unsqueeze(-1), None).squeeze(2) # B M 3
    #print(v0.shape)
    v0 = v0.unsqueeze(2).repeat(1,1,24,1)
    v1 = v1.unsqueeze(2).repeat(1,1,24,1)
    #print(v0.shape)
    v2 = v_idx
    v_idx_ = torch.reshape(v_idx,[batch_size,-1]) # B M*24  2000*24
    #print(v_idx_.dtype)
    v_idx_ = v_idx_.type(torch.int64)
    v2 = knn_gather(centers, v_idx_.unsqueeze(-1), None).squeeze(2) # B M 3
    #print(v2.shape)
    v2 = torch.reshape(v2,[batch_size,edge_num,24,3])
    #print(v2.shape)
    vec_01 = v1 - v0
    vec_02 = v2 - v0
    # B M 24 3 一个面的两个向量，可以计算法向，但是会出现
    normals = torch.cross(vec_01,vec_02,dim=3)  # B M 24 3
    # 计算每个法向与其他法向的夹角
    normals_A = normals.unsqueeze(3).repeat(1,1,1,24,1)
    normals_B = normals.unsqueeze(2).repeat(1,1,24,1,1)
    #print(normals_A[0,0,0])
    #print(normals_B[0,0,0])
    #print(normals.shape)
    len_Al2 = normals_A.pow(2).sum(-1)
    len_Bl2 = normals_B.pow(2).sum(-1)
    len_Al1 = (len_Al2 + eps).sqrt()
    len_Bl1 = (len_Bl2 + eps).sqrt()

    normals_AB = (normals_A * normals_B).sum(-1)

    cos1 = 1.0 - torch.abs(normals_AB / (len_Al1 * len_Bl1 + eps)) # B N 24 24 需要用mask去覆盖，然后加和再除以mask的和
    v_mask_init = v_mask
    v_mask = v_mask.unsqueeze(2).repeat(1,1,24,1)
    #print(v_mask.shape)
    #print(v_mask[0,0,0])
    if condition is True:
        choose_mask = torch.lt(cos1, 1-threshold)
        #print(choose_mask[0,0,1])
        v_mask = v_mask * choose_mask
        #print(v_mask[0,0,1])
        #exit()
    cos1_mask = cos1 * v_mask
    
    #print(cos1[0,0,0])
    #print(cos1.shape) 
    #print(cos1_mask[0,0,0])
    cos1_mask_average_0 = cos1_mask.sum(-1) / (v_mask.sum(-1)+eps) # B M 24
    #print(cos1_mask_average_0.shape)
    cos1_mask_average_1 = cos1_mask_average_0 * v_mask_init # B M 24
    #print(cos1_mask_average_1.shape)
    cos1_mask_average_2 = cos1_mask_average_1.sum(-1) / (v_mask_init.sum(-1)+eps)
    #print(cos1_mask_average_2.shape)
    cos1_mask_average_3 = (cos1_mask_average_2 * e_mask).sum(-1) / (e_mask.sum(-1)+eps)
    #print(cos1_mask_average_3.shape)
    flatten_loss = cos1_mask_average_3.mean()
    #print(flatten_loss)
    #exit()
    return flatten_loss


def face_regularization(spheres,faces):
    '''
    spheres B N 4
    faces B M 3
    获取面上的3个球半径，然后使得他们三个的方差最小
    '''

    s1= knn_gather(spheres, faces[:,:,0].unsqueeze(-1), None).squeeze(2)[:,:,3] # B N 3
    s2 = knn_gather(spheres, faces[:,:,1].unsqueeze(-1), None).squeeze(2)[:,:,3] # B N 3
    s3 = knn_gather(spheres, faces[:,:,2].unsqueeze(-1), None).squeeze(2)[:,:,3] # B N 3
    s_mean = (s1 + s2 + s3) / 3.0
    #print(s_mean.shape)
    s_norm = ((s1 - s_mean).pow(2) + (s2 - s_mean).pow(2) + (s3 - s_mean).pow(2)) / 3.0#.sqrt()
    loss = torch.mean(s_norm.squeeze(),dim=1)
    return loss.mean()

def face_regularization_min(spheres,finetune_spheres,faces):
    '''
    spheres B N 4
    faces B M 3
    使得每个face的球都去接近绵中最小球
    应当是在输入球中找到face的最小球
    然后fintune的球向这个靠近
    否则，如果是finetune的球，会出问题
    '''


    r1 = knn_gather(spheres, faces[:,:,0].unsqueeze(-1), None).squeeze(2)[:,:,3] # B N 3
    r2 = knn_gather(spheres, faces[:,:,1].unsqueeze(-1), None).squeeze(2)[:,:,3] # B N 3
    r3 = knn_gather(spheres, faces[:,:,2].unsqueeze(-1), None).squeeze(2)[:,:,3] # B N 3
    r_12 = torch.cat([r1.unsqueeze(-1),r2.unsqueeze(-1)],dim=-1)
    radii_face = torch.cat([r_12,r3.unsqueeze(-1)],dim=-1)

    radii_min = radii_face.min(2)[0].unsqueeze(-1)
    
    r1 = knn_gather(finetune_spheres, faces[:,:,0].unsqueeze(-1), None).squeeze(2)[:,:,3] # B N 3
    r2 = knn_gather(finetune_spheres, faces[:,:,1].unsqueeze(-1), None).squeeze(2)[:,:,3] # B N 3
    r3 = knn_gather(finetune_spheres, faces[:,:,2].unsqueeze(-1), None).squeeze(2)[:,:,3] # B N 3
    r_12 = torch.cat([r1.unsqueeze(-1),r2.unsqueeze(-1)],dim=-1)
    radii_face_finetune = torch.cat([r_12,r3.unsqueeze(-1)],dim=-1)
    radii_dis = (radii_face_finetune - radii_min).pow(2)
    #print(radii_dis.shape)
    #exit()
    loss = torch.mean(torch.sum(radii_dis,dim=2),dim=1).mean()
    return loss

def edge_regularization_near(spheres,edges):
    '''
    spheres B N 4
    faces B M 2
    使得每个face的球都去接近绵中最小球
    '''

    r1 = knn_gather(spheres, edges[:,:,0].unsqueeze(-1), None).squeeze(2)[:,:,3] # B N 3
    r2 = knn_gather(spheres, edges[:,:,1].unsqueeze(-1), None).squeeze(2)[:,:,3] # B N 3
    
    radii_dis = (r1 - r2).pow(2)
    #print(radii_dis.shape)
    loss = torch.mean(radii_dis,dim=1).mean()
    return loss


def cd_2d(gt_a,gt_b,pred):
    '''
    gt_a B 1000 2   a
    gt_b B 1000 2   b
    pred 4B 200 2   分别为 aa  ab ba bb视角 
    其中 aa ba 与 gt_a 进行cd  b对应
    '''
    batch_size = gt_a.shape[0]

    gt_a_  = gt_a / 223.0
    gt_b_ = gt_b / 223.0
    pred_ = pred / 223.0
    cd_aa,_,_ = chamfer_distance(gt_a_,pred_[:batch_size])
    cd_ba,_,_ = chamfer_distance(gt_a_,pred_[batch_size:batch_size*2])
    cd_ab,_,_ = chamfer_distance(gt_b_,pred_[batch_size*2:batch_size*3])
    cd_bb,_,_ = chamfer_distance(gt_b_,pred_[batch_size*3:])

    #cd_loss = cd_aa + cd_ab + cd_ba + cd_bb 
    cd_loss = cd_aa + cd_bb
    return cd_loss / 4.0


def chamfer_1(gt,pred):
    length_gt = torch.from_numpy(np.zeros(gt.shape[0],'int64')) + gt.shape[1]
    length_gt = length_gt.cuda()
    length_pred = torch.from_numpy(np.zeros(pred.shape[0],'int64')) + pred.shape[1]
    length_pred = length_pred.cuda()
    #print(spheres_view1.shape)
    x_nn = knn_points(gt, pred, lengths1=length_gt, lengths2=length_pred, K=1)
    y_nn = knn_points(pred, gt, lengths1=length_pred, lengths2=length_gt, K=1)
    print(x_nn.idx.shape)
    print(y_nn.idx.shape)
    # print(pred.shape)
    # print(gt.shape)
    # print(x_nn.idx.shape)
    #exit()
    nearest_p1_p2 = knn_gather(pred, x_nn.idx, None).squeeze(2)
    nearest_p2_p1 = knn_gather(gt, y_nn.idx, None).squeeze(2)
    print(nearest_p1_p2.shape)
    print(nearest_p2_p1.shape)
    exit()
    #print(nearest_p1_p2.shape)

    cd_loss = 10 * torch.mean(torch.mean(torch.norm(gt - nearest_p1_p2,dim=2),dim=1),dim=0) \
            + torch.mean(torch.mean(torch.norm(pred - nearest_p2_p1,dim=2),dim=1),dim=0)
    #print(cd_loss)
    return cd_loss



def chamfer_2(gt,pred):
    batch_size = gt.shape[0]
    length_gt = torch.from_numpy(np.zeros(gt.shape[0],'int64')) + gt.shape[1]
    length_gt = length_gt.cuda()
    length_pred = torch.from_numpy(np.zeros(pred.shape[0],'int64')) + pred.shape[1]
    length_pred = length_pred.cuda()
    #print(spheres_view1.shape)
    x_nn = knn_points(gt, pred, lengths1=length_gt, lengths2=length_pred, K=1)
    y_nn = knn_points(pred, gt, lengths1=length_pred, lengths2=length_gt, K=1)
    # x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    # y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)
    print(cham_x.shape)
    print(cham_y.shape)
    exit()
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)

    cham_x /= length_gt
    cham_y /= length_pred

    cham_x /= batch_size
    cham_y /= batch_size

    cd_loss = cham_x * 10 + cham_y
    return cd_loss

def cd_2d_1(gt_a,gt_b,pred):
    '''
    gt_a B 1000 2   a
    gt_b B 1000 2   b
    pred 4B 200 2   分别为 aa  ab ba bb视角 
    其中 aa ba 与 gt_a 进行cd  b对应
    '''
    batch_size = gt_a.shape[0]

    gt_a_  = gt_a / 223.0
    gt_b_ = gt_b / 223.0
    pred_ = pred / 223.0
    cd_aa = chamfer_1(gt_a_,pred_[:batch_size])
    cd_ba = chamfer_1(gt_a_,pred_[batch_size:batch_size*2])
    cd_ab = chamfer_1(gt_b_,pred_[batch_size*2:batch_size*3])
    cd_bb = chamfer_1(gt_b_,pred_[batch_size*3:])

    cd_loss = cd_aa + cd_ab + cd_ba + cd_bb 
    return cd_loss / 4.0# * 0.01


def texture_loss(pred,gt_a,gt_b):
    #print(pred[0][0])
    #print(gt_a[0])
    #exit()
    #print(torch.max(pred[0][:,0:3]))
    #print(torch.max(gt_a[:,0:3]))
    l1 = torch.pow(gt_a[:,0:3] - pred[0][:,0:3], 2).mean()
    l2 = torch.pow(gt_a[:,0:3] - pred[1][:,0:3], 2).mean()
    l3 = torch.pow(gt_b[:,0:3] - pred[2][:,0:3], 2).mean()
    l4 = torch.pow(gt_b[:,0:3] - pred[3][:,0:3], 2).mean()
    rgb_loss = l1+l2+l3+l4
    #print(rgb_loss)
    #exit()
    return rgb_loss


def template_loss(centers,faces):
    # centers B N 3 
    # faces B M 3 以四边形为例， ,012 230 共边02，然后计算01,03，可以计算两个面家教，
    # 也可以根据边，来约束一个边长，保证一个平均长度。
    batch_size = centers.shape[0]
    sphere_num = centers.shape[1]
    centers = torch.reshape(centers,(batch_size,sphere_num//4,4,3))
    centers_0 = centers[:,:,0,:]
    centers_1 = centers[:,:,1,:]
    centers_2 = centers[:,:,2,:]
    centers_3 = centers[:,:,3,:]
    edge_01  = centers_1 - centers_0
    edge_02  = centers_2 - centers_0
    edge_03  = centers_3 - centers_0
    edge_01 = F.normalize(edge_01,dim=2) # B N/4 3
    edge_02 = F.normalize(edge_02,dim=2)
    edge_03 = F.normalize(edge_03,dim=2)
 
    n021 = F.normalize(torch.cross(edge_01,edge_02,dim=2),dim=2) # B N/4 3 单位向量
    n023 = F.normalize(torch.cross(edge_02,edge_03,dim=2),dim=2) # B N/4 3

    normals_AB = (n021 * n023).sum(-1)
    cos1 = 1.0 - torch.abs(normals_AB) # 平行的话，接近于0度，也就是接近于1，cos1越小，说明越平行。
    # B N/4 1 
    return cos1.mean() 

class kNNLoss(nn.Module):
    """
    Proposed PatchVariance component
    """
    def __init__(self, k=10, n_seeds=20):
        super(kNNLoss,self).__init__()
        self.k = k
        self.n_seeds = n_seeds
    def forward(self, pcs):
        n_seeds = self.n_seeds
        k = self.k
        num = pcs.shape[1]
        seeds = farthest_point_sample(pcs,n_seeds) # which gives index
        seeds_value = torch.stack([pc[seed] for pc, seed in zip(pcs,seeds)]) 
        pcs_new = pcs.unsqueeze(2).repeat(1,1,n_seeds,1)
        seeds_new = seeds_value.unsqueeze(1).repeat(1,num,1,1)
        dist = pcs_new.add(-seeds_new)
        dist_value = torch.norm(dist,dim=3)
        dist_new = dist_value.transpose(1,2)
        top_dist, idx = torch.topk(dist_new, k+1, dim=2, largest=False)
        overall_mean = top_dist[:,:,1:].mean()
        top_dist = top_dist/overall_mean
        var = torch.var(top_dist.mean(dim=2)).mean()
        return var


class kNNSphereLoss(nn.Module):
    """
    Proposed PatchVariance component
    """
    def __init__(self, k=10, n_seeds=20):
        super(kNNLoss,self).__init__()
        self.k = k
        self.n_seeds = n_seeds
    def forward(self, pcs):
        n_seeds = self.n_seeds
        k = self.k
        num = pcs.shape[1]
        seeds = farthest_point_sample(pcs,n_seeds) # which gives index
        seeds_value = torch.stack([pc[seed] for pc, seed in zip(pcs,seeds)]) 
        pcs_new = pcs.unsqueeze(2).repeat(1,1,n_seeds,1)
        seeds_new = seeds_value.unsqueeze(1).repeat(1,num,1,1)
        dist = pcs_new.add(-seeds_new)
        dist_value = torch.norm(dist,dim=3)
        dist_new = dist_value.transpose(1,2)
        top_dist, idx = torch.topk(dist_new, k+1, dim=2, largest=False)
        overall_mean = top_dist[:,:,1:].mean()
        top_dist = top_dist/overall_mean
        var = torch.var(top_dist.mean(dim=2)).mean()
        return var


class SphereInterLoss(nn.Module):
    """
    Proposed PatchVariance component
    """
    def __init__(self, k=10):
        super(SphereInterLoss,self).__init__()
        self.k = k
    def forward(self, spheres):
        centers = spheres[:,:,0:3]
        radii = spheres[:,:,3].unsqueeze(-1)
        #print(radii.shape)
        length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
        length = length.cuda()
        nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=self.k+1)
        neighbor_centers_k= knn_gather(centers, nn.idx[:,:,1:], None).squeeze(2) # B N K 3
        neighbor_radii_k= knn_gather(radii, nn.idx[:,:,1:], None).squeeze(2) # B N K 1
        centers = centers.unsqueeze(2).repeat(1,1,self.k,1) # B N K 3
        dis = torch.norm(centers - neighbor_centers_k,dim=3) # B N K 
        #print(dis[0,0])
        radii = radii.unsqueeze(2).repeat(1,1,self.k,1) # B N 10 1
        #print(radii.shape)
        sphere_dis = dis - (radii + neighbor_radii_k).squeeze() # B N 10
        top_dist, idx = torch.topk(sphere_dis, 1, dim=2, largest=False) # B N 1
        #print(top_dist[0])
        #print(top_dist.shape)
        var = torch.var(top_dist,dim=1).mean()
        #print(var)
        #exit()


        return var

def farthest_point_sample(xyz, npoint):
    
    """
    code borrowed from: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]  
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
    	# Update the i-th farthest point
        centroids[:, i] = farthest
        # Take the xyz coordinate of the farthest point
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        # Calculate the Euclidean distance from all points in the point set to this farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # Update distances to record the minimum distance of each point in the sample from all existing sample points
        mask = dist < distance
        distance[mask] = dist[mask]
        # Find the farthest point from the updated distances matrix, and use it as the farthest point for the next iteration
        farthest = torch.max(distance, -1)[1]
    return centroids

def square_distance(src, dst):
    """
    code borrowed from: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2     
	     = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)) # 2*(xn * xm + yn * ym + zn * zm)
    dist += torch.sum(src ** 2, -1).view(B, N, 1) # xn*xn + yn*yn + zn*zn
    dist += torch.sum(dst ** 2, -1).view(B, 1, M) # xm*xm + ym*ym + zm*zm
    return dist

def query_ball_point(radius, xyz, new_xyz, nsample=500, density_only=True):
    """
    code borrowed from: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists: [B, S, N] Record the Euclidean distance between the center point and all points
    sqrdists = square_distance(new_xyz, xyz) # shape (B, S, N)
    # Find all distances greater than radius^2, its group_idx is directly set to N; the rest retain the original value
    
    if not density_only:
        group_idx[sqrdists > radius ** 2] = N
        # Do ascending order, the front is greater than radius^2 are N, will be the maximum, so will take the first nsample points directly in the remaining points
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        # Considering that there may be points in the previous nsample points that are assigned N (ie, less than nsample points in the spherical area), this point needs to be discarded, and the first point can be used instead.
        # group_first: [B, S, k], actually copy the value of the first point in group_idx to the dimension of [B, S, K], which is convenient for subsequent replacement.
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        # Find the point where group_idx is equal to N
        mask = group_idx == N
        # Replace the value of these points with the value of the first point
        group_idx[mask] = group_first[mask]
        return group_idx
    else:
        raw_mat = torch.zeros(B,S,N)
        density_mat = torch.zeros(B,S)
        raw_mat[sqrdists <= radius ** 2] = 1
        density_mat = torch.sum(raw_mat,2)
        # print(torch.max(sqrdists))
        return density_mat


class kNNRepulsionLoss(nn.Module):
    """
    adapted PU-Net's uniform loss
    算的是seeds邻近的斥力，而不是每个点的
    """
    def __init__(self, k=10, n_seeds=20, h=0.01):
        super(kNNRepulsionLoss,self).__init__()
        self.k = k
        self.n_seeds = n_seeds
        self.h = h
    def forward(self, pcs):
        tic = time.time()
        n_seeds = self.n_seeds
        k = self.k
        num = pcs.shape[1]
        seeds = farthest_point_sample(pcs,n_seeds) # which gives index
        temp = time.time()
        seeds_value = torch.stack([pc[seed] for pc, seed in zip(pcs,seeds)]) # grad

        pcs_new = pcs.unsqueeze(2).repeat(1,1,n_seeds,1)
        seeds_new = seeds_value.unsqueeze(1).repeat(1,num,1,1)
        dist = pcs_new.add(-seeds_new)
        dist_value = torch.norm(dist,dim=3) # B N n_seeds  每个点到M个中心点距离
        toc = time.time()
        dist_new = dist_value.transpose(1,2) # 转置 B n_seeds N
        tac = time.time()
        top_dist, idx = torch.topk(dist_new, k+1, dim=2, largest=False) # 每个seeds选取最近k个点距离
        top_dist_net = top_dist[:,:,1:]
        weights = torch.exp(-torch.pow(top_dist_net,2)*(1/(self.h**2))) # 计算每个seeds与邻近点的斥力权重
        repulsion = torch.mul(-top_dist_net,weights) # 距离x权重=斥力
        return repulsion.sum(2).sum(1).mean()


def get_fine_loss(render_images, s_m_a, s_m_b, outputs,losses_dicts,cfg,writer,i):
    rep_loss = 0.0
    s_loss = 0.0
    sphereLoss = losses_dicts['sphereLoss']
    overall_loss = 0.0
    loss = multiview_iou_loss(render_images, s_m_a, s_m_b)
    overall_loss += loss

    if cfg['fine_loss']['repusionloss']:
        rep_loss = cfg['fine_loss']['repul_w'] * repulsion_loss(outputs['centers'],outputs['radii'])
        overall_loss+=rep_loss
    if cfg['fine_loss']['sphereloss']:
        s_loss = cfg['fine_loss']['sphere_w'] * sphereLoss(outputs['spheres'])
        overall_loss+=s_loss
    if cfg['fine_loss']['nearrloss']:
        near_r_loss = cfg['fine_loss']['nearr_w'] * nearest_r_loss(outputs['centers'],outputs['radii'])
        overall_loss+=near_r_loss
    
    writer.add_scalar("Loss/mtv_fine",loss,i)
    writer.add_scalar("Loss/rep_loss",rep_loss,i)
    writer.add_scalar("Loss/near_r_fine",near_r_loss,i)
    writer.add_scalar("Loss/s_loss_fine",s_loss,i)
    return overall_loss,loss,near_r_loss,rep_loss,s_loss,writer


def get_surf_loss(render_images, s_m_a, s_m_b, outputs,losses_dicts,cfg,writer,i):
    rep_loss = 0.0
    s_loss = 0.0
    sphereLoss = losses_dicts['sphereLoss']
    overall_loss = 0.0
    loss = multiview_iou_loss(render_images, s_m_a, s_m_b)
    overall_loss += loss
    if cfg['surf_loss']['repusionloss']:
        rep_loss = cfg['surf_loss']['repul_w'] * repulsion_loss(outputs['centers'],outputs['radii'])
        overall_loss+=rep_loss
    if cfg['surf_loss']['sphereloss']:
        s_loss = cfg['surf_loss']['sphere_w'] * sphereLoss(outputs['spheres'])
        overall_loss+=s_loss
    if cfg['surf_loss']['nearrloss']:
        near_r_loss = cfg['surf_loss']['nearr_w'] * nearest_r_loss(outputs['centers'],outputs['radii'])
        overall_loss+=near_r_loss
    writer.add_scalar("Loss/mtv_surf",loss,i)
    writer.add_scalar("Loss/rep_loss_surf",rep_loss,i)
    writer.add_scalar("Loss/near_r_surf",near_r_loss,i)
    writer.add_scalar("Loss/s_loss_surf",s_loss,i)
    return overall_loss,loss,near_r_loss,rep_loss,s_loss,writer

def get_joint_loss(render_images, mask_a,mask_b, s_m_a, s_m_b, outputs,losses_dicts,cfg,writer,i):
    rep_loss = 0.0
    s_loss = 0.0
    sphereLoss = losses_dicts['sphereLoss']
    overall_loss = 0.0
    loss = multiview_iou_loss(render_images, mask_a, mask_b)
    overall_loss += loss
    if cfg['joint_loss']['fineloss']:
        fine_loss = cfg['joint_loss']['fine_w'] * multiview_iou_fine_loss(render_images,s_m_a[:,0,:,:],s_m_b[:,0,:,:],s_m_a[:,0,:,:],s_m_b[:,0,:,:])
        overall_loss += fine_loss
    if cfg['joint_loss']['repusionloss']:
        rep_loss = cfg['joint_loss']['repul_w'] * repulsion_loss(outputs['centers'],outputs['radii'])
        overall_loss += rep_loss
    if cfg['joint_loss']['sphereloss']:
        s_loss = cfg['joint_loss']['sphere_w'] * sphereLoss(outputs['spheres'])
        overall_loss += s_loss
    if cfg['joint_loss']['nearrloss']:
        near_r_loss = cfg['joint_loss']['nearr_w'] * nearest_r_loss(outputs['centers'],outputs['radii'])
        overall_loss += near_r_loss

    writer.add_scalar("Loss/mtv_joint",loss,i)
    writer.add_scalar("Loss/rep_loss_joint",rep_loss,i)
    writer.add_scalar("Loss/near_r_joint",near_r_loss,i)
    writer.add_scalar("Loss/s_loss_jiont",s_loss,i)
    return overall_loss,loss,fine_loss,near_r_loss,rep_loss,s_loss,writer


def get_finetune_loss(render_images, mask_a,mask_b, s_m_a, s_m_b, outputs,losses_dicts,cfg,writer,i):
    rep_loss = 0.0
    s_loss = 0.0
    sphereLoss = losses_dicts['sphereLoss']
    overall_loss = 0.0
    loss = multiview_iou_loss(render_images, mask_a, mask_b)
    overall_loss += loss
    if cfg['joint_loss']['fineloss']:
        fine_loss = cfg['joint_loss']['fine_w'] * multiview_iou_fine_loss(render_images,s_m_a[:,0,:,:],s_m_b[:,0,:,:],s_m_a[:,0,:,:],s_m_b[:,0,:,:])
        overall_loss += fine_loss
    if cfg['joint_loss']['nearrloss']:
        near_r_loss = cfg['joint_loss']['nearr_w'] * nearest_r_loss(outputs['centers'],outputs['radii'])
        overall_loss += near_r_loss

    writer.add_scalar("Loss/mtv_finetune",loss,i)
    writer.add_scalar("Loss/fine_finetune",fine_loss,i)
    writer.add_scalar("Loss/nearr_finetune",near_r_loss,i)
    return overall_loss,loss,fine_loss,near_r_loss,writer

# class kNNRepulsionLoss(nn.Module):
#     """
#     adapted PU-Net's uniform loss
#     """
#     def __init__(self, k=10, n_seeds=20, h=0.01):
#         super(kNNRepulsionLoss,self).__init__()
#         self.k = k
#         self.n_seeds = n_seeds
#         self.h = h
#     def forward(self, centers):
#         tic = time.time()
#         length = torch.from_numpy(np.zeros(centers.shape[0],'int64')) + centers.shape[1]
#         length = length.cuda()
#         nn = knn_points(centers[:,:,0:3], centers[:,:,0:3], lengths1=length, lengths2=length, K=self.k+1)

#         nearest_centers = knn_gather(centers, nn.idx[:,:,1:], None).squeeze(2) # B N K 3

#         #top_dist, idx = torch.topk(dist_new, k+1, dim=2, largest=False)
#         #top_dist_net = top_dist[:,:,1:]
#         dis = torch.norm(centers.unsqueeze(2).repeat(1,1,self.k,1) - nearest_centers,dim=3) # B N K 3
#         #print(dis.shape)
#         weights = torch.exp(-torch.pow(dis,2)*(1/(self.h**2)))
#         repulsion = torch.mul(-dis,weights)
#         return repulsion.sum(2).sum(1).mean()




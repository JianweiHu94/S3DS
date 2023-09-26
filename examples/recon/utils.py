import torch
import numpy as np
import numpy as np 
import torch 
import utils_hole_fill
import cv2
import kaolin.io.obj as kobj 
import os 
import sys 
import yaml

# General config
def load_config(cfg_file):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    with open(cfg_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)


    return cfg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Mesh(object):
    def __init__(self,vertices,faces):
        self.vertices = torch.from_numpy(vertices)
        self.faces = torch.from_numpy(faces)



def img_cvt(images):
    return (255. * images).detach().cpu().numpy().clip(0, 255).astype('uint8').transpose(1, 2, 0)


def img_cvt_merge(gt_mask,render_mask):
    #print(gt_mask.shape)
    #print(render_mask.shape)
    gt = (255. * gt_mask).detach().cpu().numpy().clip(0, 255).astype('uint8').transpose(1, 2, 0) # N N 3 (0~255)
    render = (255.*render_mask).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0) #N N 3 (0~1)
    idx = np.array(np.where(render[:,:,3] > 0),'int').transpose()
    gt[idx[:,0],idx[:,1],1] = 0
    gt[idx[:,0],idx[:,1],2] = 0
    gt[idx[:,0],idx[:,1],0] = 255
    gt[idx[:,0],idx[:,1],3] = 255
    
    return gt


def write_ma(filename,spheres):
    #print(valid_sphere.shape)
    np.savetxt(filename,spheres)


def write_cam_info(filename,cam_info):
    pass


def load_obj_spherenum(obj_file,sphere_num):
    
    mesh = kobj.import_mesh(obj_file,False)
    vertices = mesh.vertices 
    faces = mesh.faces 
    vertices = vertices.cpu().detach().numpy()
    faces = faces.cpu().detach().numpy()
    v_num = vertices.shape[0]
    f_num = faces.shape[0]

    vertices_many = np.expand_dims(vertices,axis=0).repeat(sphere_num,axis=0)

    vertices_many = np.reshape(vertices_many,[sphere_num*v_num,3])

    faces_many = np.expand_dims(faces,axis=0).repeat(sphere_num,axis=0)
    faces_many = np.reshape(faces_many,[sphere_num*f_num,3])
    for i in range(int(sphere_num)):
        faces_many[i*f_num:(i+1)*f_num,:] += v_num * i


    return Mesh(vertices_many,faces_many)

def transfer_spheres(spheres):
    sphere_num = spheres.shape[0]
    mesh = kobj.import_mesh('./data/obj/sphere/sphere_42.obj',False)
    vertices = mesh.vertices 
    faces = mesh.faces 
    vertices = vertices.cpu().detach().numpy()
    faces = faces.cpu().detach().numpy()
    v_num = vertices.shape[0]
    f_num = faces.shape[0]

    vertices_many = np.expand_dims(vertices,axis=0).repeat(sphere_num,axis=0)

    vertices_many = np.reshape(vertices_many,[sphere_num*v_num,3])

    faces_many = np.expand_dims(faces,axis=0).repeat(sphere_num,axis=0)
    faces_many = np.reshape(faces_many,[sphere_num*f_num,3])
    for i in range(int(sphere_num)):
        faces_many[i*f_num:(i+1)*f_num,:] += v_num * i
    
    vertices = torch.from_numpy(vertices_many).cuda()
    faces = torch.from_numpy(faces_many).cuda()
    bias = torch.unsqueeze(spheres[:,0:3],axis=1).repeat(1,42,1)
    radii = torch.unsqueeze(spheres[:,3],axis=1).unsqueeze(1).repeat(1,42,1)
    bias = torch.reshape(bias,(1,-1,3))
    radii = torch.reshape(radii,(1,-1,1)).repeat(1,1,3)

    base = torch.unsqueeze(vertices,0).repeat(1,1,1)
    base = base * radii + bias
    return base[0],faces


def export_ma_off(off_file,spheres,faces):
    #print(faces.shape)
    real_faces = []
    #faces = list(set(list(faces)))
    for face in faces:
        face = list(face)
        if face not in real_faces:
            real_faces.append(face)
    utils_hole_fill.write_off(off_file,spheres,[],real_faces)


def save_2d_png(demo_skeletons,skeleton_file):
    test_mask = np.zeros((224,224,3),'float')
    for k in range(demo_skeletons.shape[0]):
        test_mask[int(demo_skeletons[k,0]),int(demo_skeletons[k,1]),:] = 255.0 
    cv2.imwrite(skeleton_file, test_mask)


def load_template(filename,num):
    fin = open(filename)
    lines = open(filename, "r").read().replace("\\n", " ").splitlines()
    sphere_num = int(lines[0].split()[0])
    face_num = int(lines[0].split()[1])
    centers = []
    faces = []
    for i in range(1, sphere_num + 1):
        centers.append([float(s) for s in lines[i].split()[0:3]])
    for i in range(sphere_num + 1, sphere_num + face_num + 1):
        faces.append([int(f) for f in lines[i].split()[0:3]])
    #print(faces)
    centers_all = [] 
    faces_all=[]
    for i in range(num):
        for j in range(len(centers)):
            centers_all.append([centers[j][0],centers[j][1],centers[j][2]])
        for j in range(len(faces)):
            faces_all.append([faces[j][0]+i*4,faces[j][1]+i*4,faces[j][2]+i*4])
    centers_all = np.array(centers_all,'float32')
    faces_all = np.array(faces_all,'int')
    #print(centers_all.shape)
    #print(faces_all.shape)
    #print(faces_all[0:8])
    return centers_all, faces_all
    

def save_obj(filename, vertices, faces):
    assert vertices.ndimension() == 2
    assert faces.ndimension() == 2

    faces = faces.detach().cpu().numpy()

    with open(filename, 'w') as f:
        f.write('# %s\n' % os.path.basename(filename))
        f.write('#\n')
        f.write('\n')

        for vertex in vertices:
            f.write('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
        f.write('\n')

        for face in faces:
            f.write('f %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))
    f.close()

import os

#import soft_renderer.functional as srf
import torch
import numpy as np
import tqdm
from torchvision.transforms import Resize
import cv2
import torchvision.transforms as transforms
from PIL import Image
import time
import threading
import queue
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import camera_util



transform_list = [
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

image_to_tensor = transforms.Compose(transform_list)


class_ids_map = {
    '02691156': 'Airplane',
    '02828884': 'Bench',
    '02933112': 'Cabinet',
    '02958343': 'Car',
    '03001627': 'Chair',
    '03211117': 'Display',
    '03636649': 'Lamp',
    '03691459': 'Loudspeaker',
    '04090263': 'Rifle',
    '04256520': 'Sofa',
    '04379243': 'Table',
    '04401088': 'Telephone',
    '04530566': 'Watercraft',
}


class_sample_num_map = {
    '02691156': 2831,
    '02828884': 1271,
    '02933112': 1100,
    '02958343': 5247,
    '03001627': 4744,
    '03211117': 766,
    '03636649': 1622,
    '03691459': 1132,
    '04090263': 1660,
    '04256520': 2221,
    '04379243': 5956,
    '04401088': 736,
    '04530566': 1357,
}


DATA_DIR = '/home2/hujianwei/Dataset/mesh_reconstruction/viewangle_%d_224_split/'

class ShapeNet(Dataset):
    def __init__(self,class_id=None, set_name=None):
        self.class_id = class_id 
        self.set_name = set_name
        #self.view30_dir = DATA_DIR%(30) + '/03001627/03001627/'
        #self.view0_dir = DATA_DIR%(0) + '/03001627/03001627/'
        fin = open('./data/data_list/%s_%s_cat.txt'%(class_id,set_name),'r')
        self.datalist = [x.rsplit() for x in fin.readlines()]
        self.real_len =  len(self.datalist)

        print(len(self.datalist))
        self.elevation = 30.
        self.distance = 2.732


    def __len__(self):
        return 99999999

    def __getitem__(self,index):
        index = index % self.real_len
        line = self.datalist[index]

        cat_id,idx1,idx2,view1,view2 = line
        view1_id = float(idx1.split('_')[1])
        view2_id = float(idx2.split('_')[1])
        dir1 = DATA_DIR%int(view1) + '/%s/'%(cat_id)
        dir2 = DATA_DIR%int(view2) + '/%s/'%(cat_id)

        img1_init = cv2.imread(dir1 + '/images/%s.png'%idx1,cv2.IMREAD_UNCHANGED)
        img2_init = cv2.imread(dir2 + '/images/%s.png'%idx2,cv2.IMREAD_UNCHANGED)

        img1 = np.array(img1_init,'float32') / 255.
        img2 = np.array(img2_init,'float32') / 255.

        mask1 = img1[:,:,3]
        mask2 = img2[:,:,3]

        fine1 = cv2.imread(dir1 + '/fine/%s.png'%idx1,cv2.IMREAD_UNCHANGED)
        fine2 = cv2.imread(dir2 + '/fine/%s.png'%idx2,cv2.IMREAD_UNCHANGED)
        fine1 =  np.array(fine1,'float32') / 255.
        fine2 =  np.array(fine2,'float32') / 255.
       
        tensor1 = image_to_tensor(Image.fromarray(img1_init[:,:,0:3]))
        tensor2 = image_to_tensor(Image.fromarray(img2_init[:,:,0:3]))
        viewpoints_1 = camera_util.get_points_from_angles(self.distance, float(view1), -view1_id * 15)
        viewpoints_1 = np.array(viewpoints_1,'float32')
     
        viewpoints_2 = camera_util.get_points_from_angles(self.distance, float(view2), -view2_id * 15)
        viewpoints_2 = np.array(viewpoints_2,'float32')

        return img1,img2,viewpoints_1,viewpoints_2,mask1,mask2,tensor1,tensor2,fine1,fine2



def get_eval_img():
    eval_img_init = cv2.imread('/home2/hujianwei/Dataset/mesh_reconstruction/viewangle_30_224_split_test/03001627/0047_01.png',cv2.IMREAD_UNCHANGED) 
    eval_img = np.array(eval_img_init,'float32') / 255.
    tensor1 = image_to_tensor(Image.fromarray(eval_img_init[:,:,0:3]))
    tensor1 = tensor1.unsqueeze(0).cuda() # 1 3 224 224
    viewpoints = camera_util.get_points_from_angles(2.732, float(30), -1.0 * 15)
    viewpoints = np.array(viewpoints,'float32')
    viewpoints = torch.from_numpy(viewpoints).cuda()
    mask1 = eval_img[:,:,3]
    return eval_img,mask1,tensor1,viewpoints


        
if __name__ == '__main__':
    
    CLASS_IDS_ALL = ('03001627')
    t = ShapeNet(CLASS_IDS_ALL.split(',')[0], 'train')
    dl = DataLoader(t,batch_size=64,num_workers=16,shuffle=False)
    #start = time.time()
    for batch in dl:
        # print(time.time()-start)
        # start = time.time()
        img1,img2,viewpoints_1,viewpoints_2,mask1,mask2,tensor1,tensor2,fine1,fine2 = batch 
        print(img1.shape)
        print(viewpoints_1.shape)
        print(mask1.shape)
        print(tensor1.shape)
        print(fine1.shape)
        exit()


    
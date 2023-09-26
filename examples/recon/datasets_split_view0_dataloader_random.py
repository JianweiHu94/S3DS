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
import random


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


DATA_DIR = './data/shapenet_img/viewangle_%d_224_split/'

class ShapeNet(Dataset):
    def __init__(self,class_id=None, set_name=None):
        self.class_id = class_id 
        self.set_name = set_name
        if self.class_id is None:
            fin = open('./data/data_list/allcats_train.txt','r')
        else:
            fin = open('./data/data_list/train/cats_%s_%s.txt'%(class_id,set_name),'r')
        self.datalist = [x.rsplit() for x in fin.readlines()]
        self.real_len =  len(self.datalist)

        print(len(self.datalist))
        self.elevation = 30.
        self.distance = 2.732
        self.angle = [0,30]
        


    def __len__(self):
        return 99999999

    def __getitem__(self,index):
        #index = index % self.real_len
        idx = random.randint(0,self.real_len-1)
        line = self.datalist[idx]

        cat_id,sample_id = line
        idx1 = random.randint(0,23)
        idx2 = random.randint(0,23)
        view1 = self.angle[random.randint(0,1)]
        view2 = self.angle[random.randint(0,1)]
        view1_id = float(idx1)
        view2_id = float(idx2)
        dir1 = DATA_DIR%int(view1) + '/%s/'%(cat_id)
        dir2 = DATA_DIR%int(view2) + '/%s/'%(cat_id)
        #print(dir1 + '/images/%s_%02d.png'%(sample_id,idx1))

        img1_init = cv2.imread(dir1 + '/images/%s_%02d.png'%(sample_id,idx1),cv2.IMREAD_UNCHANGED)
        img2_init = cv2.imread(dir2 + '/images/%s_%02d.png'%(sample_id,idx2),cv2.IMREAD_UNCHANGED)

        img1 = np.array(img1_init,'float32') / 255.
        img2 = np.array(img2_init,'float32') / 255.

        mask1 = img1[:,:,3]
        mask2 = img2[:,:,3]

        fine1 = cv2.imread(dir1 + '/fine/%s_%02d.png'%(sample_id,idx1),cv2.IMREAD_UNCHANGED)
        fine2 = cv2.imread(dir2 + '/fine/%s_%02d.png'%(sample_id,idx2),cv2.IMREAD_UNCHANGED)
        fine1 =  np.array(fine1,'float32') / 255.
        fine2 =  np.array(fine2,'float32') / 255.
       
        tensor1 = image_to_tensor(Image.fromarray(img1_init[:,:,0:3]))
        tensor2 = image_to_tensor(Image.fromarray(img2_init[:,:,0:3]))
        viewpoints_1 = camera_util.get_points_from_angles(self.distance, float(view1), -view1_id * 15)
        viewpoints_1 = np.array(viewpoints_1,'float32')
     
        viewpoints_2 = camera_util.get_points_from_angles(self.distance, float(view2), -view2_id * 15)
        viewpoints_2 = np.array(viewpoints_2,'float32')

        return img1,img2,viewpoints_1,viewpoints_2,mask1,mask2,tensor1,tensor2,fine1,fine2



        
if __name__ == '__main__':
    
    t = ShapeNet(None, None)
    dl = DataLoader(t,batch_size=4,num_workers=4,shuffle=False)

    for batch in dl:

        img1,img2,viewpoints_1,viewpoints_2,mask1,mask2,tensor1,tensor2,fine1,fine2 = batch 
 
        exit()




    
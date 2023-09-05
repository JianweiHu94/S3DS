# S3DS: Self-supervised Learning of 3D Skeletons from Single View Images

This is the code of the paper " S3DS: Self-supervised Learning of 3D Skeletons from Single View Images".

## Requirements
python 3.7.12
torch 1.11.0+cu113
tensorboard 2.11.0
pytorch3d 0.7.1 
kaolin 0.12.0

## Datasets

https://drive.google.com/file/d/1IeB9dshm6ET23Appc9MHaLlq40Vj4cAZ/view?usp=sharing   about 3.4G

Download the shapenet_img to S3DS/data/


Download resnet18-5c106cde.pth to S3DS/checkpoint/

## Training & Evaluation
cd S3DS
python examples/recon/train.py -eid 03001627 -config ./configs/default_03001627.yaml

python examples/recon/test.py -cls 03001627 -eid 03001627 -md ./data/results/models/03001627/checkpoint_finetune_0100000.pth.tar
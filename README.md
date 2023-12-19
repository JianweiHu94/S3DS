# S3DS: Self-supervised Learning of 3D Skeletons from Single View Images

This is the code of the paper " S3DS: Self-supervised Learning of 3D Skeletons from Single View Images".


## Eviroments
conda create -n S3DS python=3.7.12

conda activate S3DS

pip install -r requirements.txt

pip install kaolin==0.12.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.11.0_cu113.html

cd pytorch3d

python setup.py install

## Datasets

https://drive.google.com/file/d/1IeB9dshm6ET23Appc9MHaLlq40Vj4cAZ/view?usp=sharing   about 3.4G

Download the shapenet_img to S3DS/data/


## Training & Evaluation
cd S3DS
python examples/recon/train.py -eid 03001627 -config ./configs/default_03001627.yaml

python examples/recon/test.py -cls 03001627 -eid 03001627 -md ./data/results/models/03001627/checkpoint_finetune_0100000.pth.tar
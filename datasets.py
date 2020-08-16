import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image

import random
import json
import cv2
import numpy as np
import scipy.io as scio

def make_dataset(root,status):
    # os.path.splitext() 函数将文件名和扩展名分开
    # 获得NLPR和NJU2000的根目录
    NLPR_root = root.split('_')[0] + root.split('_')[1]
    nju2000_root = root.split('_')[0] + root.split('_')[2]
    # NLPR_root = root.split('_')[0] + 'NLPR/'
    # nju2000_root = root.split('_')[0] + 'NJU2000/'
    nlpr_img = NLPR_root + 'RGB/'
    nlpr_trth = NLPR_root + 'groundtruth/'
    nlpr_depth = NLPR_root + 'hha/'
    nju2000_img = nju2000_root + 'LR/'
    nuj2000_trth = nju2000_root + 'GT/'
    nuj2000_depth = nju2000_root + 'hha/'
    nlpr_list_path = NLPR_root + 'nlpr_' + status + '.json'
    nju2000_list_path = nju2000_root + 'nju2000_' + status + '.json'
    # 加载训练集
    with open(nlpr_list_path) as f1:
        nlpr_list = json.load(f1)
    with open(nju2000_list_path) as f2:
        nju2000_list = json.load(f2)
    nlpr_temp = [(os.path.join(nlpr_img, img_name + '.jpg'), os.path.join(nlpr_trth, img_name + '.jpg'),os.path.join(nlpr_depth, img_name + '_Depth.jpg')) for img_name in nlpr_list]
    nju2000_temp = [(os.path.join(nju2000_img, img_name + '.jpg'), os.path.join(nuj2000_trth, img_name + '.png'),os.path.join(nuj2000_depth, img_name + '.jpg')) for img_name in nju2000_list]
    train_dataset = nlpr_temp
    train_dataset.extend(nju2000_temp)
    # stere_list = os.listdir('./data/STERE_train/RGB/')
    # stere_temp = [(os.path.join('./data/STERE_train/RGB', img_name), os.path.join('./data/STERE_train/GT', img_name.replace('.jpg','.png')),os.path.join('./data/STERE_train/hha', img_name)) for img_name in stere_list]
    # train_dataset.extend(stere_temp)
    return train_dataset


class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, status='train',joint_transform=None, transform=None, target_transform=None, depth_transform=None):
        self.root = root
        self.imgs = make_dataset(root,status)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.depth_transform = depth_transform

    def __getitem__(self, index):
        img_path, gt_path,depth_path = self.imgs[index]
        img_name = gt_path.split('/')[4]
        rgb_img = Image.open(img_path).convert('RGB')
        depth_img = Image.open(depth_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            rgb_img, target,depth_img = self.joint_transform(rgb_img, target,depth_img)
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
            depth_img = self.depth_transform(depth_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return rgb_img,depth_img,target, img_name

    def __len__(self):
        return len(self.imgs)


def make_dataset1(root,status):
    # os.path.splitext() 函数将文件名和扩展名分开
    # 获得NLPR和NJU2000的根目录
    NLPR_root = root.split('_')[0] + root.split('_')[1]
    nju2000_root = root.split('_')[0] + root.split('_')[2]
    nlpr_img = NLPR_root + 'RGB/'
    nlpr_trth = NLPR_root + 'groundtruth/'
    nlpr_depth = NLPR_root + 'hha/'
    nju2000_img = nju2000_root + 'LR/'
    nuj2000_trth = nju2000_root + 'GT/'
    nuj2000_depth = nju2000_root + 'hha/'
    nlpr_list_path = NLPR_root + 'nlpr_' + status + '.json'
    nju2000_list_path = nju2000_root + 'nju2000_' + status + '.json'
    # 加载训练集
    with open(nlpr_list_path) as f1:
        nlpr_list = json.load(f1)
    with open(nju2000_list_path) as f2:
        nju2000_list = json.load(f2)
    nlpr_temp = [(os.path.join(nlpr_img, img_name + '.jpg'), os.path.join(nlpr_trth, img_name + '.jpg'),os.path.join(nlpr_depth, img_name + '_Depth.jpg')) for img_name in nlpr_list]
    nju2000_temp = [(os.path.join(nju2000_img, img_name + '.jpg'), os.path.join(nuj2000_trth, img_name + '.png'),os.path.join(nuj2000_depth, img_name + '.jpg')) for img_name in nju2000_list]
    train_dataset = nlpr_temp
    train_dataset.extend(nju2000_temp)
    # LFSD data root
    lfsd_temp = [(('./data/LFSD/all_focus_images/'+i+'.jpg'),('./data/LFSD/ground_truth/'+i+'.png'),('./data/LFSD/hha/'+i+'.jpg')) for i in os.listdir('./data/LFSD/all_focus_images')]
    train_dataset.extend(lfsd_temp)
    return train_dataset

class ImageFolder1(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, status='train',joint_transform=None, transform=None, target_transform=None, depth_transform=None):
        self.root = root
        self.imgs = make_dataset(root,status)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.depth_transform = depth_transform

    def __getitem__(self, index):
        img_path, gt_path,depth_path = self.imgs[index]
        img_name = gt_path.split('/')[4]
        rgb_img = Image.open(img_path).convert('RGB')
        depth_img = Image.open(depth_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            rgb_img, target,depth_img = self.joint_transform(rgb_img, target,depth_img)
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
            depth_img = self.depth_transform(depth_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return rgb_img,depth_img,target, img_name

    def __len__(self):
        return len(self.imgs)

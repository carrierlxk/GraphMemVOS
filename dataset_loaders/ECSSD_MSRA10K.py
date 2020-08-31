import torch
import torch.utils.data as data

import os, math, random
from os.path import join
import numpy as np

import cv2
from .custom_transforms_MTB import aug_batch
from PIL import Image
import matplotlib.pyplot as plt


class ECSSD(data.Dataset):
    def __init__(self, root='', replicates=1, aug=False):
        self.replicates = replicates
        self.aug = aug

        image_root = join(root, 'images')
        gt_root = join(root, 'saliencymaps')

        self.image_list = []
        self.gt_list = []
        files = sorted(os.listdir(image_root))
        for i in range(len(files)):
            img = join(image_root, files[i])
            gt = join(gt_root, files[i][:-4] + '.png')
            self.image_list += [img]
            self.gt_list += [gt]

        self.size = len(self.image_list)
        self.frame_size = cv2.imread(self.image_list[0], cv2.IMREAD_COLOR).shape

        assert (len(self.image_list) == len(self.gt_list))

    def __getitem__(self, index):

        index = index % self.size
        img = cv2.imread(self.image_list[index], cv2.IMREAD_COLOR)

        gt = np.array(cv2.imread(self.gt_list[index], cv2.IMREAD_GRAYSCALE))#np.expand_dims(, axis=2) Image.open(self.gt_list[index])
        gt[gt == 255] = 1

        image_size = img.shape[:2]

        images = []
        segannos = []
        for i in range(3):
            img_copy, gt_copy = np.copy(img), np.copy(gt)
            img_auged, gt_auged = aug_batch(img_copy, gt_copy)
            img_auged = img_auged.transpose(2, 0, 1)
            gt_auged = gt_auged.transpose(2, 0, 1)
            img_auged = torch.from_numpy(img_auged.astype(np.float32))
            gt_auged = torch.from_numpy(gt_auged.astype(np.float32))
            images.append(img_auged)
            segannos.append(gt_auged)

        images = torch.stack(images, dim=0).float().clamp(0, 1)
        segannos = torch.stack(segannos, dim=0).float()

        num_objects = int(segannos.max())
        # save sample for checking, todo: need to delete
        if False:
            path_sample = '/home/cgv841/gwb/Code/agame-vos-master/ECSSD_sample/{}'.format(index)
            if not os.path.exists(path_sample):
                os.makedirs(path_sample)
            palette = Image.open(
                '/home/cgv841/gwb/DataSets/davis-2017/data/DAVIS/Annotations/480p/blackswan/00000.png').getpalette()
            for i in range(images.shape[0]):
                img, gt = 255 * images[i], segannos[i]
                img, gt = img.numpy().transpose((1, 2, 0)).astype(np.uint8), gt.numpy().transpose((1, 2, 0)).astype(
                    np.uint8).squeeze()
                img, gt = Image.fromarray(img), Image.fromarray(gt)
                gt.putpalette(palette)
                img.save(os.path.join(path_sample, '{:05d}.jpg'.format(i)))
                gt.save(os.path.join(path_sample, '{:05d}.png'.format(i)))

        return {'images':images, 'segannos':segannos, 'seqname':'unknow', 'num_objects':num_objects}

    def __len__(self):
        return self.size * self.replicates


class MSRA10K(data.Dataset):
    def __init__(self, root='', replicates=1, aug=False):
        self.replicates = replicates
        self.aug = aug

        image_root = join(root, 'images')
        gt_root = join(root, 'saliencymaps')

        self.image_list = []
        self.gt_list = []
        files = sorted(os.listdir(image_root))
        for i in range(len(files)):
            img = join(image_root, files[i])
            gt = join(gt_root, files[i][:-4] + '.png')
            self.image_list += [img]
            self.gt_list += [gt]

        self.size = len(self.image_list)
        self.frame_size = cv2.imread(self.image_list[0], cv2.IMREAD_COLOR).shape

        assert (len(self.image_list) == len(self.gt_list))

    def __getitem__(self, index):

        index = index % self.size
        img = cv2.imread(self.image_list[index], cv2.IMREAD_COLOR)

        gt = np.array(cv2.imread(self.gt_list[index], cv2.IMREAD_GRAYSCALE))#, cv2.IMREAD_GRAYSCALE np.expand_dims(, axis=2) Image.open(self.gt_list[index])
        #print('gt size:',gt.shape)
        #np.array(Image.open(self.gt_list[index]))#np.expand_dims(, axis=2)
        gt[gt != 255] = 0
        gt[gt == 255] = 1

        image_size = img.shape[:2]

        images = []
        segannos = []
        for i in range(3):
            img_copy, gt_copy = np.copy(img), np.copy(gt)
            img_auged, gt_auged = aug_batch(img_copy, gt_copy)
            img_auged = img_auged.transpose(2, 0, 1)
            gt_auged = gt_auged.transpose(2, 0, 1)
            img_auged = torch.from_numpy(img_auged.astype(np.float32))
            gt_auged = torch.from_numpy(gt_auged.astype(np.float32))
            images.append(img_auged)
            segannos.append(gt_auged)

        images = torch.stack(images, dim=0).float().clamp(0, 1)
        segannos = torch.stack(segannos, dim=0).float()

        num_objects = int(segannos.max())
        # save sample for checking, todo: need to delete
        if False:
            path_sample = '/home/ubuntu/xiankai/STM_train_v1/MSRA10K_sample/{}'.format(index)
            if not os.path.exists(path_sample):
                os.makedirs(path_sample)
            palette = Image.open(
                '/raid/DAVIS/DAVIS-2017/DAVIS-train-val/Annotations/480p/blackswan/00000.png').getpalette()
            for i in range(images.shape[0]):
                img, gt = 255 * images[i], segannos[i]
                img, gt = img.numpy().transpose((1, 2, 0)).astype(np.uint8), gt.numpy().transpose((1, 2, 0)).astype(
                    np.uint8).squeeze()
                img, gt = Image.fromarray(img), Image.fromarray(gt)
                gt.putpalette(palette)
                img.save(os.path.join(path_sample, '{:05d}.jpg'.format(i)))
                gt.save(os.path.join(path_sample, '{:05d}.png'.format(i)))

        return {'images': images, 'segannos': segannos, 'seqname': 'unknow', 'num_objects': num_objects}

    def __len__(self):
        return self.size * self.replicates
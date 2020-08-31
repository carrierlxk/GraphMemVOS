from PIL import Image
import torch
from pycocotools.coco import COCO
import os
import numpy as np
import random
import torch
import scipy.io as scio
from torchvision import transforms
from.custom_transforms_MTB import aug_batch

class SimpleCoCoDataset(torch.utils.data.Dataset):
    def __init__(self, rootdir, set_name='val2017', transform=None, max_num_objects=3):
        self.rootdir, self.set_name = rootdir, set_name
        self.transform = transform
        self.coco = COCO(os.path.join(self.rootdir, 'annotations', 'instances_'
                                      + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.load_classes()
        self._max_num_objects = max_num_objects

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        # coco ids is not from 1, and not continue
        # make a new index from 0 to 79, continuely

        # classes:             {names:      new_index}
        # coco_labels:         {new_index:  coco_index}
        # coco_labels_inverse: {coco_index: new_index}
        self.classes, self.coco_labels, self.coco_labels_inverse = {}, {}, {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # labels:              {new_index:  names}
        self.labels = {}
        for k, v in self.classes.items():
            self.labels[v] = k

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img, gt = self.load_image_anns(index)

        sample = {'image': img, 'gt': gt}

        images = []
        segannos = []
        for i in range(3):
            sample_copy = {'image': np.copy(sample['image']), 'gt': np.copy(sample['gt'])}
            sample_transformed = self.transform(sample_copy)
            images.append(sample_transformed['crop_image'])
            segannos.append(sample_transformed['crop_gt'])

        images = torch.stack(images, dim=0).float().clamp(0, 1)
        segannos = torch.stack(segannos, dim=0).float()
        num_objects = int(segannos.max())
        #print('coco size:',images.size(), segannos.size())
        # save sample for checking, todo: need to delete
        if False:
            path_coco_sample = '/raid/STM_train_v1/coco_sample/{}'.format(index)
            if not os.path.exists(path_coco_sample):
                os.makedirs(path_coco_sample)
            palette = Image.open(
                '/raid/DAVIS/DAVIS-2017/DAVIS-train-val/Annotations/480p/blackswan/00000.png').getpalette()
            for i in range(images.shape[0]):
                img, gt = 255*images[i], segannos[i]
                img, gt = img.numpy().transpose((1, 2, 0)).astype(np.uint8), gt.numpy().transpose((1, 2, 0)).astype(np.uint8).squeeze()
                img, gt = Image.fromarray(img), Image.fromarray(gt)
                gt.putpalette(palette)
                img.save(os.path.join(path_coco_sample, '{:05d}.jpg'.format(i)))
                gt.save(os.path.join(path_coco_sample, '{:05d}.png'.format(i)))

        return {'images':images, 'segannos':segannos, 'seqname':'unknow', 'num_objects':num_objects}

    def load_image_anns(self, index):
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        imgpath = os.path.join(self.rootdir, self.set_name,
                               image_info['file_name'])
        img = np.array(Image.open(imgpath).convert('RGB'))/255.

        annotation_ids = self.coco.getAnnIds(self.image_ids[index], iscrowd=False)
        coco_anns = self.coco.loadAnns(annotation_ids)
        target = np.zeros([img.shape[0], img.shape[1]]).astype(np.float32)
        if len(coco_anns) == 0:
            return img, target
        coco_anns_sample = random.sample(coco_anns, min(self._max_num_objects, len(coco_anns)))
        for i, a in enumerate(coco_anns_sample):
            target[self.coco.annToMask(a) == 1] = i+1
        target = target[:, :, np.newaxis]

        return img, target

    def image_aspect_ratio(self, index):
        image = self.coco.loadImgs(self.image_ids[index])[0]
        return float(image['width']) / float(image['height'])

class SimpleSBDDataset(torch.utils.data.Dataset):
    def __init__(self, rootdir, transform=None, max_num_objects=3):
        self.rootdir = rootdir
        self.transform = transform
        self.image_list = []
        self.gt_list = []
        files = sorted(os.listdir(rootdir + '/img'))
        for i in range(len(files)):
            img = os.path.join(rootdir + '/img', files[i])
            gt = os.path.join(rootdir + '/inst', files[i][:-4] + '.mat')
            self.image_list += [img]
            self.gt_list += [gt]

        self._max_num_objects = max_num_objects

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img, gt = self.load_image_anns(index)

        sample = {'image': img, 'gt': gt}

        images = []
        segannos = []
        for i in range(3):
            sample_copy = {'image': np.copy(sample['image']), 'gt': np.copy(sample['gt'])}
            sample_transformed = self.transform(sample_copy)
            images.append(sample_transformed['crop_image'])
            segannos.append(sample_transformed['crop_gt'])

        images = torch.stack(images, dim=0).float().clamp(0, 1)
        segannos = torch.stack(segannos, dim=0).float()
        num_objects = int(segannos.max())
        # save sample for checking, todo: need to delete
        if False:
            path_coco_sample = '/home/cgv841/gwb/Code/agame-vos-master/SBD_sample/{}'.format(index)
            if not os.path.exists(path_coco_sample):
                os.makedirs(path_coco_sample)
            palette = Image.open('/home/cgv841/gwb/DataSets/davis-2017/data/DAVIS/Annotations/480p/blackswan/00000.png').getpalette()
            for i in range(images.shape[0]):
                img, gt = 255*images[i], segannos[i]
                img, gt = img.numpy().transpose((1, 2, 0)).astype(np.uint8), gt.numpy().transpose((1, 2, 0)).astype(np.uint8).squeeze()
                img, gt = Image.fromarray(img), Image.fromarray(gt)
                gt.putpalette(palette)
                img.save(os.path.join(path_coco_sample, '{:05d}.jpg'.format(i)))
                gt.save(os.path.join(path_coco_sample, '{:05d}.png'.format(i)))

        return {'images':images, 'segannos':segannos, 'seqname':'unknow', 'num_objects':num_objects}

    def load_image_anns(self, index):
        imgpath = self.image_list[index]
        img = np.array(Image.open(imgpath).convert('RGB'))/255.

        gtpath = self.gt_list[index]
        gt = scio.loadmat(gtpath)
        gt = gt['GTinst']['Segmentation'][0][0]

        old_idx_list = random.sample(range(1, gt.max()+1), min(self._max_num_objects, gt.max()))
        target = np.zeros([img.shape[0], img.shape[1]]).astype(np.float32)
        for i, old_idx in enumerate(old_idx_list):
            target[gt == old_idx] = i+1

        target = target[:, :, np.newaxis]

        return img, target

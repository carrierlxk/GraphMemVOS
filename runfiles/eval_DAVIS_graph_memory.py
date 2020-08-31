from __future__ import division
import sys
sys.path.append('/media/xiankai/Data/segmentation/OSVOS/ECCV_graph_memory')
import models
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
#from torchvision import models
from numpy.random import randint
# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import csv

### My libs

from dataset import DAVIS_MO_Test


torch.set_grad_enabled(False)  # Volatile
def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size #input size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array

def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", default='0')
    parser.add_argument("-c", type=str, help="checkpoint", default=' ')
    parser.add_argument("-s", type=str, help="set", default="val")
    parser.add_argument("-y", type=int, help="year", default="17")
    parser.add_argument("-viz", help="Save visualization", action="store_true")
    parser.add_argument("-D", type=str, help="path to data", default='/media/xiankai/Data/segmentation/DAVIS-2017/DAVIS-train-val')
    return parser.parse_args()

args = get_arguments()

GPU = args.g
YEAR = args.y
SET = args.s
VIZ = args.viz
DATA_ROOT = args.D

# Model and version
MODEL = 'Graph-memory'
print(MODEL, ': Testing on DAVIS')

os.environ['CUDA_VISIBLE_DEVICES'] = GPU
if torch.cuda.is_available():
    print('using Cuda devices, num:', torch.cuda.device_count())

if VIZ:
    print('--- Produce mask overaid video outputs. Evaluation will run slow.')
    print('--- Require FFMPEG for encoding, Check folder ./viz')

palette = Image.open('/media/xiankai/Data/segmentation/DAVIS-2017/DAVIS-train-val/Annotations/480p/bear/00000.png').getpalette()


class VideoRecord(object):
    pass


def _sample_pair_indices(record):
    """
    :param record: VideoRecord
    :return: list
    """
    new_length = 1

    average_duration = (record.num_frames - new_length + 1) // record.num_segment
    if average_duration > 0:
        # offsets = np.multiply(list(range(record.num_segment)), average_duration) + randint(average_duration,size=record.num_segment)
        offsets = np.multiply(list(range(record.num_segment)), average_duration) + [average_duration//2]*record.num_segment  # no random
    elif record.num_frames > record.num_segment:
        offsets = randint(record.num_frames -
                          new_length + 1, size=record.num_segment)
    else:
        offsets = np.zeros((record.num_segment,))
    return offsets


def Run_video(Fs, Ms, num_frames, num_objects, Mem_every=None, Mem_number=None):
    # initialize storage tensors
    num_first_memory = 1
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in
                       np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]  # [0, 5, 10, 15, 20, 25]
    else:
        raise NotImplementedError

    # print('memory size:', len(to_memorize))
    Es = torch.zeros_like(Ms)  # mask
    Es[:, :, 0] = Ms[:, :, 0]
    record = VideoRecord()

    record.num_segment = 4
    for t in tqdm.tqdm(range(1, num_frames)):
        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(Fs[:, :, t - 1], Es[:, :, t - 1], torch.tensor([num_objects]))

        if t - 1 == 0:  #
            this_keys, this_values = prev_key, prev_value  # only prev memory
        elif t <= record.num_segment:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        # segment
        with torch.no_grad():
            # print('input size1:', this_keys.size(), this_values.size())# torch.Size([1, 11, 128, 1, 30, 57]) torch.Size([1, 11, 512, 1, 30, 57]) # one hot label vector with length 11
            record.num_frames = t
            select_keys = []
            select_values = []
            # print('key size:',t,this_keys.size(), this_values.size())#[1, 11, 128, 4, 30, 57]
            if t > record.num_segment:
                Index = _sample_pair_indices(record) if record.num_segment else []

                #Index[-1]=t-1
                # print('index', t, Index, type(Index))
                for add_0 in range(num_first_memory):
                    select_keys.append(this_keys[:, :, :, 0, :, :].unsqueeze(dim=3))
                    select_values.append(this_values[:, :, :, 0, :, :].unsqueeze(dim=3))
                # print('index0:', this_keys[:, :, :, 0, :, :].size(),this_values[:, :, :, 0, :, :].unsqueeze(dim=3).size())
                for ii in Index:
                    prev_key1, prev_value1 = model(Fs[:, :, ii], Es[:, :, ii], torch.tensor([num_objects]))

                    # print('index1:', prev_key.size()) #1, 11, 128, 1, 30, 57
                    select_keys.append(prev_key1)  # (this_keys[:, :, :, t-1, :, :])
                    select_values.append(prev_value1)  # (this_values[:, :, :, t-1, :, :])
                select_keys.append(prev_key)  # (this_keys[:, :, :, t-1, :, :])
                select_values.append(prev_value)
                # print('index2:', select_keys[0].size())
                select_keys = torch.cat(select_keys, dim=3)
                select_values = torch.cat(select_values, dim=3)
                # print('key size:', prev_key.size(), select_keys.size(), select_values.size())
            else:
                select_keys = this_keys
                select_values = this_values
            logit = model(Fs[:, :, t], select_keys, select_values, torch.tensor([num_objects]))

        Es[:, :, t] = F.softmax(logit, dim=1)
        # print('output size:', torch.max(Es[:,:,t]), torch.min(Es[:,:,t]))
        # update
        if t - 1 in to_memorize:
            keys, values = this_keys, this_values

    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    return pred, Es


Testset = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR, SET), single_object=(YEAR == 16))
Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model = nn.DataParallel(models.graph_memory())
for param in model.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    model.cuda()
model.eval()  # turn-off BN

pth_path = args.c
print('Loading weights:', pth_path)

checkpoint = torch.load(pth_path)
model.module.load_state_dict(checkpoint['net'])

try:
    print('epoch:', checkpoint['epoch'])
except:
    print('dont know epoch')

for seq, V in enumerate(Testloader):
    Fs, Ms, num_objects, info = V
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()
    print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0][0]))

    B, K, N, H, W = Fs.shape
    H_1, W_1 = 480, int(480.0 * W / H)
    # resize_sizes = [(int(0.75 * H_1), int(0.75 * W_1)), (H_1, W_1), (int(1.25 * H_1), int(1.25 * W_1))]
    resize_sizes = [(H_1, W_1)]
    use_flip = True
    resize_Fs = []
    resize_Ms = []
    # ms
    for size in resize_sizes:
        resize_Fs.append(F.interpolate(input=Fs.squeeze(0).permute(1, 0, 2, 3), size=size, mode='bilinear',
                                       align_corners=True).permute(1, 0, 2, 3).unsqueeze(0))
        resize_Ms.append(F.interpolate(input=Ms.squeeze(0).permute(1, 0, 2, 3), size=size, mode='nearest'
                                       ).permute(1, 0, 2, 3).unsqueeze(0))
    # flip
    if use_flip:
        for i in range(len(resize_Fs)):
            resize_Fs.append(torch.flip(resize_Fs[i], [-1]))
            resize_Ms.append(torch.flip(resize_Ms[i], [-1]))

    Es_list = []
    for i in range(len(resize_Fs)):
        pred, Es = Run_video(resize_Fs[i], resize_Ms[i], num_frames, num_objects, Mem_every=5, Mem_number=None)
        Es = F.interpolate(input=Es.squeeze(0).permute(1, 0, 2, 3), size=(H, W), mode='bilinear',
                           align_corners=True).permute(1, 0, 2, 3).unsqueeze(0)
        if use_flip:
            if i >= (len(resize_Fs) / 2):
                Es = torch.flip(Es, [-1])
        Es_list.append(Es)
    Es = torch.stack(Es_list).mean(dim=0)
    pred = np.argmax(Es[0].numpy(), axis=0).astype(np.uint8)  # different than ytvos here
    # pred, Es = Run_video(Fs, Ms, num_frames, num_objects, Mem_every=1, Mem_number=None,
    #                      num_first_memory=num_first_memory, num_middle_memory=num_middle_memory)

    # Save results for quantitative eval ######################
    test_path = os.path.join('./test', code_name, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for f in range(num_frames):
        img_E = Image.fromarray(pred[f])
        # print('image size:',type(YEAR),np.max(pred[f]),np.min(pred[f]))
        if YEAR == 16:
            # print('ok!')
            img_E = (pred[f].squeeze() * 255).astype(np.uint8)
            img_E = Image.fromarray(img_E)
            img_E = img_E.convert('RGB')
        else:
            img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))

    if VIZ:
        from tools.helpers import overlay_davis

        # visualize results #######################
        viz_path = os.path.join('./viz/', code_name, seq_name)
        if not os.path.exists(viz_path):
            os.makedirs(viz_path)

        for f in range(num_frames):
            pF = (Fs[0, :, f].permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
            pE = pred[f]
            canvas = overlay_davis(pF, pE, palette)
            canvas = Image.fromarray(canvas)
            canvas.save(os.path.join(viz_path, 'f{}.jpg'.format(f)))

        vid_path = os.path.join('./viz/', code_name, '{}.mp4'.format(seq_name))
        frame_path = os.path.join('./viz/', code_name, seq_name, 'f%d.jpg')



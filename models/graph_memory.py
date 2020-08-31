from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
 
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
import sys

from .helpers import *
 
print('Space-time Memory Networks: initialized.')
import sys
sys.path.append('/home/ubuntu/xiankai/meta_VOS/models')
import units.ConvGRU2 as ConvGRU

class ASPP(nn.Module):
    def __init__(self, dilation_series, padding_series, depth):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(1024, depth, 1, 1)
        self.bn_x = nn.BatchNorm2d(depth)
        self.conv2d_0 = nn.Conv2d(1024, depth, kernel_size=1, stride=1)
        self.bn_0 = nn.BatchNorm2d(depth)
        self.conv2d_1 = nn.Conv2d(1024, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                  dilation=dilation_series[0])
        self.bn_1 = nn.BatchNorm2d(depth)
        self.conv2d_2 = nn.Conv2d(1024, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                  dilation=dilation_series[1])
        self.bn_2 = nn.BatchNorm2d(depth)
        self.conv2d_3 = nn.Conv2d(1024, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                  dilation=dilation_series[2])
        self.bn_3 = nn.BatchNorm2d(depth)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = nn.Conv2d(depth * 5, depth, kernel_size=3, padding=1)  # 512 1x1Conv
        #self.bn = nn.BatchNorm2d(depth)
        #self.prelu = nn.PReLU()
        # for m in self.conv2d_list:
        #    m.weight.data.normal_(0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_stage_(self, dilation1, padding1):
        Conv = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=padding1, dilation=dilation1, bias=True)  # classes
        Bn = nn.BatchNorm2d(256)
        Relu = nn.ReLU(inplace=True)
        return nn.Sequential(Conv, Bn, Relu)

    def forward(self, x):
        # out = self.conv2d_list[0](x)
        # mulBranches = [conv2d_l(x) for conv2d_l in self.conv2d_list]
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = self.bn_x(image_features)
        image_features = self.relu(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear', align_corners=True)
        out_0 = self.conv2d_0(x)
        out_0 = self.bn_0(out_0)
        out_0 = self.relu(out_0)
        out_1 = self.conv2d_1(x)
        out_1 = self.bn_1(out_1)
        out_1 = self.relu(out_1)
        out_2 = self.conv2d_2(x)
        out_2 = self.bn_2(out_2)
        out_2 = self.relu(out_2)
        out_3 = self.conv2d_3(x)
        out_3 = self.bn_3(out_3)
        out_3 = self.relu(out_3)
        out = torch.cat([image_features, out_0, out_1, out_2, out_3], 1)
        out = self.bottleneck(out)
        # out = self.bn(out)
        # out = self.prelu(out)
        # for i in range(len(self.conv2d_list) - 1):
        #    out += self.conv2d_list[i + 1](x)

        return out

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 

class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_o):
        #print('type:',type(self.mean),type(in_f))

        f = (in_f - self.mean) / self.std
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float() # add channel dim

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_o(o) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f
 
class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        #print('fea size:',s.size(),pm.size())
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.layer5 = self._make_pred_layer(ASPP, [2, 4, 8], [2, 4, 8], mdim)
        #self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.layer5(r4))
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        
        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p #, p2, p3, p4

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        self.propagate_layers = 5
        self.conv_fusion = nn.Conv2d(512, 512, kernel_size=1, bias=True)
        self.ConvGRU_h = ConvGRU.ConvGRUCell(512, 512, all_dim=128, kernel_size=1)
        self.ConvGRU_m = ConvGRU.ConvGRUCell(512, 512, all_dim=128, kernel_size=1)
        self.linear_e = nn.Linear(128, 128, bias=False)
 
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()  # T is the memory size
        _, D_o, _, _, _ = m_out.size()
        q_out0 = q_out.clone()
        for kk in range(0, self.propagate_layers):
            B, D_e, T, H, W = m_in.size()
            _, D_o, _, _, _ = m_out.size()
            # print('fea size:',m_in.size(),q_in.size())
            mi = m_in.view(B, D_e, T * H * W)
            mi = torch.transpose(mi, 1, 2)  # b, THW, emb

            qi = q_in.view(B, D_e, H * W)  # b, emb, HW

            p = torch.bmm(mi, qi)  # b, THW, HW
            p = p / math.sqrt(D_e)
            p = F.softmax(p, dim=1)  # b, THW, HW
            mo = m_out.view(B, D_o, T * H * W)
            mem = torch.bmm(mo, p)  # Weighted-sum B, D_o, HW
            mem_mean = mem.view(B, D_o, H, W)

            if T <2:

                mem_out = torch.cat([mem_mean, q_out0], dim=1)
                return mem_out,10
            if T>1:
                m_in_all = torch.cat((m_in, q_in.unsqueeze(2)),dim=2).contiguous() # B, D_e, T+1, H, W
                m_out_all = torch.cat((m_out, q_out.unsqueeze(2)),dim=2).contiguous() # B, D_o, T+1, H, W
                #print('memory size:',m_in_all.size(),m_out_all.size())
                edge_featuress = []
                for x in range(0,T+1): #for each node
                    edge_set = [] #compute edge feature with other nodes
                    for y in range(0,T+1):
                        edge_feature = self.generate_edge(m_in_all[:,:,y,:,:].clone(), m_out_all[:,:,y,:,:].clone(), m_in_all[:,:,x,:,:].clone())
                        edge_set.append(edge_feature)
                    edge_set.pop(x) #remove self connection
                    edge_features = self.conv_fusion(torch.sum(torch.stack(edge_set,dim=1),dim=1))#self.conv_fusion(torch.cat(edge_set,dim=1))
                    edge_featuress.append(edge_features)
                for x in range(0,T):
                     #only update memory node
                        #print('feature dim:',torch.cat(edge_set,dim=1).size(),m_out_all[:,:,x,:,:].size())
                    hiden_state = self.ConvGRU_m(edge_featuress[x], m_out_all[:,:,x,:,:].clone())
                    #hiden_state = self.batch_norm_m(hiden_state)
                    m_out_all[:, :, x, :, :] = hiden_state

            q_out_h = self.ConvGRU_h(q_out, mem_mean)
            #q_out_h = self.batch_norm_h(q_out_h)
            q_out = q_out_h.clone()

        mem_out = torch.cat([q_out, q_out0], dim=1)
        return mem_out,10

    def generate_edge(self, m_in, m_out, q_in):
        B, D_e,  H, W = m_in.size()  # during training T is 1 or 2
        _, D_o, _, _ = m_out.size()
        mi = m_in.view(B, D_e,  H * W)
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb

        qi = q_in.view(B, D_e, H * W)  # b, emb, HW
        mi = self.linear_e(mi)
        p = torch.bmm(mi, qi)  # b, THW, HW
        #p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1)  # b, THW, HW

        mo = m_out.view(B, D_o,  H * W)
        mem = torch.bmm(mo, p)  # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)
        return mem

class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
 
    def forward(self, x):  
        return self.Key(x), self.Value(x)




class graph_memory(nn.Module):
    def __init__(self):
        super(graph_memory, self).__init__()
        self.Encoder_M = Encoder_M() 
        self.Encoder_Q = Encoder_Q() 

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(256)
 
    def Pad_memory(self, mems, num_objects, K):
        pad_mems = []
        for mem in mems:
            pad_mem = ToCuda(torch.zeros(1, K, mem.size()[1], 1, mem.size()[2], mem.size()[3]))
            pad_mem[0,1:num_objects+1,:,0] = mem
            pad_mems.append(pad_mem)
        return pad_mems

    def memorize(self, frame, masks, num_objects):
        B_f_batch, B_m_batch, B_o_batch = [], [], []
        frame_batch, masks_batch, num_objects_batch = frame, masks, num_objects
        for i in range(num_objects_batch.shape[0]):
            # memorize a frame
            frame, masks = frame_batch[i].unsqueeze(0), masks_batch[i].unsqueeze(0)
            num_objects = num_objects_batch[i]
            _, K, H, W = masks.shape # B = 1

            (frame, masks), pad = pad_divide_by([frame, masks], 16, (frame.size()[2], frame.size()[3]))

            # make batch arg list
            B_list = {'f':[], 'm':[], 'o':[]}
            for o in range(1, num_objects+1): # 1 - no
                B_list['f'].append(frame)
                B_list['m'].append(masks[:,o])
                B_list['o'].append( (torch.sum(masks[:,1:o], dim=1) + \
                    torch.sum(masks[:,o+1:num_objects+1], dim=1)).clamp(0,1) )

            # make Batch
            B_ = {}
            for arg in B_list.keys():
                B_[arg] = torch.cat(B_list[arg], dim=0)
            B_f_batch.append(B_['f']), B_m_batch.append(B_['m']), B_o_batch.append(B_['o'])

        B_f, B_m, B_o = torch.cat(B_f_batch, dim=0), torch.cat(B_m_batch, dim=0), torch.cat(B_o_batch, dim=0)
        r4, _, _, _, _ = self.Encoder_M(B_f, B_m, B_o)
        k4, v4 = self.KV_M_r4(r4) # num_objects, 128 and 512, H/16, W/16
        k4, v4 = self.Pad_memory([k4, v4], num_objects=torch.sum(num_objects_batch), K=K*num_objects_batch.shape[0])
        return k4, v4

    def Soft_aggregation(self, ps, K):
        num_objects, H, W = ps.shape
        em = ToCuda(torch.zeros(1, K, H, W)) 
        em[0,0] =  torch.prod(1-ps, dim=0) # bg prob
        em[0,1:num_objects+1] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em)))
        return logit

    def segment(self, frame, keys, values, num_objects): 
        k4e_batch, v4e_batch = [], []
        r3e_batch, r2e_batch = [], []
        frame_batch, num_objects_batch = frame, num_objects
        _, K, keydim, T, H, W = keys.shape # B = 1
        # pad


        for i in range(num_objects_batch.shape[0]):
            frame = frame_batch[i].unsqueeze(0)
            [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))
            num_objects = num_objects_batch[i]
            r4, r3, r2, _, _ = self.Encoder_Q(frame)
            k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16
            
            # expand to ---  no, c, h, w
            k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1) 
            r3e, r2e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1)
            k4e_batch.append(k4e), v4e_batch.append(v4e)
            r3e_batch.append(r3e), r2e_batch.append(r2e)
        
        k4e, v4e = torch.cat(k4e_batch, dim=0), torch.cat(v4e_batch, dim=0)
        r3e, r2e = torch.cat(r3e_batch, dim=0), torch.cat(r2e_batch, dim=0)
        # memory select kv:(1, K, C, T, H, W)
        m4, viz = self.Memory(keys[0,1:torch.sum(num_objects_batch)+1], values[0,1:torch.sum(num_objects_batch)+1], k4e, v4e)
        logits_batch = self.Decoder(m4, r3e, r2e)
        logits_batch_out = []
        begin = 0
        for i in range(num_objects_batch.shape[0]):
            ps = F.softmax(logits_batch[begin:begin + num_objects_batch[i]], dim=1)[:,1] # no, h, w  
            #ps = indipendant possibility to belong to each object
            
            logit = self.Soft_aggregation(ps, int(K/num_objects_batch.shape[0])) # 1, K, H, W
            logits_batch_out.append(logit)
            begin += num_objects_batch[i]

        logit = torch.cat(logits_batch_out, dim=0)
        if pad[2]+pad[3] > 0:
            logit = logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            logit = logit[:,:,:,pad[0]:-pad[1]]

        return logit

    def forward(self, *args, **kwargs):
        if args[1].dim() > 4: # keys
            return self.segment(*args, **kwargs)
        else:
            return self.memorize(*args, **kwargs)



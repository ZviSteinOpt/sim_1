# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:11:46 2021

@author: tzvis
"""

import torch
from PIL import Image
import scipy.io
import numpy as np 
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import models
from torch.autograd import Variable
from drawnow import drawnow, figure
import torch.nn.functional as F


class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)



net = models.resnet50(pretrained=True)
im = Image.open("C:/Users/tzvis/Desktop/dlwpt-code-master/data/p1ch2/bobby.jpg")


def  TargetF_new(im,net):
    
    
    tran = transforms.Compose([
    transforms.ToTensor()
    ])
    
    # camera_spec
    im_s = im.size
    N_r     = im_s[0]*0+720
    t_read  = 10*10**-6 #sec
    t_exp   = 35*t_read #sec
    frame_r = 30 #sec^-1
    dt_rst  = 1/(N_r*frame_r)
    # 
    mat = scipy.io.loadmat('C:/Users/tzvis/Desktop/09 these/matlab code/simulation/laser_pulse.mat')
    laser_puls = list(mat.items())
    laser_puls = laser_puls[3]
    laser_puls = laser_puls[1]
    p_laser_puls = Image.fromarray(laser_puls.astype('uint8'), 'RGB')
    p_laser_puls = tran(p_laser_puls)
    red_laser    = torch.randn(int(N_r+(t_exp/t_read)), requires_grad = True)
    #net.eval()
    #score = torch.nn.functional.softmax(net(batch_t), dim=1)[0] * 100
    #print(score)
    #percentage_np = score.detach().numpy()
    #plt.plot(percentage_np.reshape(1000,1))
    
    l = torch.zeros(1000)
    loss= torch.zeros(1000)
    l_t = np.arange(1,1000)
    for t in l_t:
        #gr, loss[t],l[t] = FGSM(tran(im),p_laser_puls,net,0,red_laser)
        red_laser.requires_grad = False
        red_laser = red_laser#-0.01*gr
        red_laser.requires_grad = True
        #plt.plot(red_laser.detach().numpy())
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(loss[1:t].detach().numpy(), 'g--')
        ax2.plot(l[1:t].detach().numpy(), 'b-')
        plt.show
        plt.grid()
        plt.pause(0.05)

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 07:29:33 2021

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


net = models.resnet50(pretrained=True)
im = Image.open("C:/Users/tzvis/Desktop/dlwpt-code-master/data/p1ch2/bobby.jpg")


def  TargetF_new(im,net):
    
    
    tran = transforms.Compose([
    transforms.ToTensor()
    ])
    
    # camera_spec
    im_s = im.size
    N_r     = im_s[0]*0+720
    t_exp   = 25*10**-6 #sec
    t_read  = 5*10**-6 #sec
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
    
    l_t = np.arange(1,100)
    for t in l_t:
        gr = FGSM(tran(im),p_laser_puls,net,0,red_laser)
        red_laser.requires_grad = False
        red_laser = red_laser+gr
        red_laser.requires_grad = True
        plt.plot(red_laser.detach().numpy())
        
def shutter(t_read,t_exp,N_r,f,ran):
    g = torch.arange(1., N_r+1)
    g = g-1
    g = g*t_read
    gg = g+t_read+t_exp
    l = torch.zeros(N_r)
    #f_phase[0:ran] = f[len(red_laser)-ran+1:len(red_laser)]
    #f_phase[ran+1:len(f)] = f[1:len(f)-ran];
    l_t = np.arange(1,(N_r+(t_exp/t_read)))
    for t in l_t:
        on = (gg>(t*t_read))*(g<(t*t_read))
        l = l+(on*0.5*(torch.tanh(f[int(t)])+1))
    return l


def FGSM(Img,laser_pulse,net,L,red_laser):
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

    r_img = torch.unsqueeze(preprocess(Img),0)
    score_r = torch.nn.functional.softmax(net(r_img), dim=1)[0]

    t_exp   = 25*10**-6 #sec
    t_read  = 5*10**-6 #sec
    N_r     = 720 #sec
    p       = 2
    j       = 1
    
    l       = shutter(t_read,t_exp,N_r,red_laser,0)/(t_exp/t_read)
    lp      = laser_pulse*l.view(720,1)
    p_image = torch.unsqueeze(preprocess(Img+lp),0)
    net.eval()
    score = torch.nn.functional.softmax(net(p_image), dim=1)[0]
    loss_f = torch.nn.CrossEntropyLoss()
    obj = sum(red_laser)+loss_f(score.view(1,1000),torch.argmax(score_r).view(-1))
    obj.backward()
    return red_laser.grad
    #percentage_np = score.detach().numpy()
    #plt.plot(percentage_np.reshape(1000,1))


    
TargetF_new(im,net)

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
import torch.nn as nn
from config import cfg
import torch
import torch.nn.functional as F
import os
from utils.math_utils import max2d, min2d

# Global Definition
device = cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu'
criterion = nn.BCELoss(weight=None,size_average=False)
CE = nn.CrossEntropyLoss()



def adjust_learning_rate(cfg, iteration, optim_rgb, optim_focal, optim_clstm, optim_intergration, epoch):

    lr = cfg.SOLVER.LR[iteration] * (1 - epoch / (cfg.SOLVER.EPOCHS+1)) ** 0.95
    param_list = [optim_rgb.param_groups,optim_focal.param_groups,
                optim_clstm.param_groups,optim_intergration.param_groups]
    for param in param_list:
        for param_group in param:
            param_group['lr'] = lr
    return lr



def Check_point(epoch,iteration,model_rgb, model_focal, model_clstm, model_intergration):
    
    dic = {'rgb':model_rgb,'focal':model_focal,
        'clstm':model_clstm,'intergration':model_intergration}
        
    for name in dic:
        save_name = ('%s/%s_snapshot_%d.pth' % (os.path.join(cfg.SAVE_ROOT,cfg.SYSTEM.EXP_NAME),name, epoch + iteration * cfg.SOLVER.EPOCHS))#+cfg.SYSTEM.ITERATION))#writer_idx+1+cfg.SYSTEM.ITERATION))
        torch.save(dic[name].state_dict(), save_name)
        print('save: (snapshot_%s: %d)' % (name, epoch + iteration * cfg.SOLVER.EPOCHS))
    return 



def BCE(output, target):
    # output [b,2,h,w]  target [b,1,h,w]
    loss = 0
    for i in range(cfg.SOLVER.NUM_MAPS): 
        loss +=  criterion(output, target)
    loss /= cfg.SOLVER.NUM_MAPS 
    return loss



def sigmoid(pred):
    '''
    input： (1,1,256,256)
    '''
    pred = torch.sigmoid(pred)
    pred = pred[:, 1, :, :]
    pred = pred[np.newaxis, ...]
    return pred



def update_acc(acc,out1,out2,y,delta):
    '''
    update slice 1,3 by compute |y-out|
    update slice 0,2 by copied 1,3 
    :param acc: numpy, shape [4,256,256] 
    :param out1: tensor [1,2,256,256], corresponding acc[0:2,:,:]
    :param out2: tensor [1,2,256,256], corresponding acc[2:4,:,:]
    return: updated acc numpy
    '''
    #update slice 0,2
    acc[0,:,:],acc[2,:,:] = acc[1,:,:],acc[3,:,:]
    #update slice 1,3
    y,out1,out2 = torch.squeeze(y),torch.squeeze(out1),torch.squeeze(out2)
    y, out1, out2 = y.data.cpu().numpy(), out1.data.cpu().numpy(), out2.data.cpu().numpy()
   
    acc[1,:,:] = np.where(np.abs(y-out1) > delta, 0, 1)
    acc[3,:,:] = np.where(np.abs(y-out2) > delta, 0, 1)
    return acc



def update_forget(acc,forget):
    '''
    update forget matrix by acc
    :param acc: numpy, shape [4,256,256]
    :param forget: numpy, shape [2,256,256] 
    :return: updated forget matrix, numpy shape [2,256,256]
    '''
    transitions1 = acc[1,:,:] - acc[0,:,:]
    transitions2 = acc[3,:,:] - acc[2,:,:]
    forget[0][np.where(transitions1 == -1 )[0],np.where(transitions1 == -1)[1]] += 1 
    forget[1][np.where(transitions2 == -1 )[0],np.where(transitions2 == -1)[1]] += 1 
    return forget



def update_M(forget):
    '''
    convert forget matrix to confidence matrix M
    :param forget: numpy shape [2,256,256]
    :return: M1, M2  tensor shape [256,256]
    '''
    M1 = 2/(1 + np.exp(cfg.SOLVER.A*(forget[0,:,:]**2)))
    M2 = 2/(1 + np.exp(cfg.SOLVER.A*(forget[1,:,:]**2)))
    M1, M2 = torch.from_numpy(M1).float().to(device),torch.from_numpy(M2).float().to(device)
    return M1, M2


def correlation_samples(loader,model):
    model_rgb, model_focal, model_clstm, model_intergration = model
    peer_iter = iter(loader)
    x1 = peer_iter.next()['image'].to(device)
    fo1 = peer_iter.next()['focal'].to(device)
    y_noise1 = peer_iter.next()['noisy_label'].to(device)
    y_n_min, y_n_max = min2d(y_noise1), max2d(y_noise1)
    y_noise1 = (y_noise1 - y_n_min) / (y_n_max - y_n_min)
    #y_noise1 = discretize_pseudolabels(y_noise1, Disc_Thr)

    basize, dime, height, width = fo1.size()  
    fo1 = fo1.view(1, basize, dime, height, width).transpose(0, 1)  
    fo1 = torch.cat(torch.chunk(fo1, 12, dim=2), dim=1)  
    fo1 = torch.cat(torch.chunk(fo1, basize, dim=0), dim=1).squeeze()  
    f1,f2,f3,f4,f5 = model_focal(fo1)
    r1,r2,r3,r4,r5 = model_rgb(x1)

    outf1, _, _, outr1, _, _ = model_clstm(r1, r2, r3, r4, r5, f1, f2, f3, f4, f5) 
                    
    out1 = model_intergration(outf1,outr1)
    out1 = sigmoid(out1).float()
    return out1


# The weight of peer term
def f_alpha(epoch):
    
    alpha = np.linspace(0.0, cfg.SOLVER.ALPHA, num=30)
    
    return alpha[epoch]



def imsave(file_name, img, img_size):
    """
    save a torch tensor as an image
    :param file_name: 'image/folder/image_name'
    :param img: 3*h*w torch tensor
    :return: nothing
    """
    assert(type(img) == torch.FloatTensor,
           'img must be a torch.FloatTensor')
    plt.imsave(file_name, img, cmap='gray')  



def Discretize(In, a):
    return (In>a)



def ThresholdPrediction(pred, target, Disc_Thr):
    #Prepare pred for thresholding
    t_up=(target>Disc_Thr).int()
    t_low=(target<Disc_Thr).int()
    a1=2
    a0=1
    #Tensor of shape of pred. Value of -1 means 'entry is original entry from pred', a1-1 means 'value should be 1', a0-1 means 'value should be 0'
    Z=a1*(pred>target).int()*t_up + a0*(pred<target).int()*t_low-1
    return (Z==-1).float()*pred + Z.float().clamp(0,1)



def discretize_pseudolabels( pseudolabels, Disc_Thr):   
    for dummy_ind in range(len(pseudolabels[0])):
            pseudolabels[0][dummy_ind] = Discretize(pseudolabels[0][dummy_ind], Disc_Thr).float()
    return pseudolabels #[b,h,w] [b,num,h,w]



def discretize_depthlab(depthlab, Disc_Thr):

    depthlab = Discretize(depthlab, Disc_Thr).float()
    return depthlab #[b,h,w] [b,num,h,w]




















#omitted code
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    input = input.transpose(1,2).transpose(2,3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def cross_entropy2d_ori(input, target, weight=None, size_average=True):  
    n, c, h, w = input.size()
    #input = input.transpose(1,2).transpose(2,3).contiguous()
    #input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = torch.squeeze(input)
    target = torch.squeeze(target)
    input = input.view(h*w , c)
    target = target.view(-1)
    #input = input.view(-1)
    # target: (n*h*w,)
    mask = target >= 0
    #target = target[mask]
    loss = CE(input, target)
    if size_average:
        loss /= mask.data.sum()
    return loss

def F_cont(sal_pred, Disc_Label, b=1.5):
    
    assert sal_pred.shape==Disc_Label.shape
    #Get True Positives, False Positives, True Negatives (Continuous!)
    TP=sal_pred*Disc_Label
    FP=sal_pred*(1-Disc_Label)
    TN=(1-sal_pred)*Disc_Label 
    #sum up TP,FP, for each image
    if int(torch.__version__[0])>0:
        TP=torch.sum(TP, dim=(1,2))
        FP=torch.sum(FP, dim=(1,2))
        TN=torch.sum(TN, dim=(1,2))
    else: #the above does not work in torch 0.4, which we need for ffi for Deeplab
        TP=torch.sum(torch.sum(TP, dim=2), dim=1)
        FP=torch.sum(torch.sum(FP, dim=2), dim=1)
        TN=torch.sum(torch.sum(TN, dim=2), dim=1)
    eps=1e-5
    prec=TP/(TP+FP+eps)
    recall=TP/(TP+TN+eps)
 
    F=(1+b)*prec*recall/(b*prec+recall+eps)
    Loss=1-F
    return torch.mean(Loss)

def compute_loss(sal_pred, pseudolabels, beta=1.0):
    loss=0.0
    sal_pred_list=[]
    pseudolabels = pseudolabels[0]
    pseudolabels = torch.unsqueeze(pseudolabels,dim=1)
    for dummy_ind in range(len(pseudolabels[0])): 
        sal_pred_list.append(sal_pred)
    for dummy_ind in range(len(sal_pred_list)):
        loss += F_cont(sal_pred_list[dummy_ind], pseudolabels[dummy_ind], b=beta)
    loss/=len(sal_pred_list)
    return loss








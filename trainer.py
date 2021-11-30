import torch
from config import cfg
from utils.Utils import *
from utils.math_utils import max2d, min2d
import os
import scipy.io as sio


def train_epoch(cfg,epoch,loader,logger,writer,Disc_Thr,optim_rgb,optim_focal,optim_clstm,optim_intergration,model_rgb, model_focal, model_clstm, model_intergration):
    
        train_loader, correlate_loader1, correlate_loader2, correlate_loader3= loader
        h, w = cfg.SOLVER.IMG_SIZE[0], cfg.SOLVER.IMG_SIZE[1]
        device = cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu'
        
        print('epoch {}'.format(epoch))
        
            
        model_rgb.train()
        model_focal.train()
        model_clstm.train()
        model_intergration.train()

        for batch_idx, data in enumerate(train_loader):
            
                #writer_idx = batch_idx * cfg.SOLVER.BATCH_SIZE + ((epoch-1) * num_batches * cfg.SOLVER.BATCH_SIZE) + 0 #cfg.SYSTEM.ITERATION
                x = data['image'].to(device)  
                fo = data['focal'].to(device)
                d = data['depth'].to(device)
                l = data['lab'].to(device)
                img_name = data['img_name']
                d = discretize_depthlab(d, 0.7)
                l = discretize_depthlab(l, Disc_Thr)

                y_noise = data['noisy_label'].to(device)
                y_n_min, y_n_max = min2d(y_noise), max2d(y_noise)
                y_noise = (y_noise - y_n_min) / (y_n_max - y_n_min)
                #y_noise = discretize_pseudolabels(y_noise, Disc_Thr)
                basize, dime, height, width = fo.size()  
                fo = fo.view(1, basize, dime, height, width).transpose(0, 1)  
                fo = torch.cat(torch.chunk(fo, 12, dim=2), dim=1)  
                fo = torch.cat(torch.chunk(fo, basize, dim=0), dim=1).squeeze()  
                f1,f2,f3,f4,f5 = model_focal(fo)
                r1,r2,r3,r4,r5 = model_rgb(x)

                outf, _, dfeature_map, outr, _, lfeature_map = model_clstm(r1, r2, r3, r4, r5, f1, f2, f3, f4, f5) 
                
                # read trans_acc,update transform matrix
                Acc_file = os.path.join(cfg.DATA.TRAIN.ROOT,'Acc_data/Acc_dir',img_name[0] + '.mat')
                Forget_file = os.path.join(cfg.DATA.TRAIN.ROOT,'Forget_data/Forget_dir',img_name[0] + '.mat')
                trans = sio.loadmat(Acc_file)
                trans = trans['acc']
                trans = update_acc(trans,sigmoid(outf), sigmoid(outr), y_noise, cfg.SOLVER.DELTA)
                sio.savemat(Acc_file, {'trans':trans})

                # compute forget matrix,save mat 
                forget = sio.loadmat(Forget_file)
                forget = forget['forget']
                forget = update_forget(trans,forget)
                sio.savemat(Forget_file, {'forget':forget})

                # compute M matrix to use 
                M1,M2 = update_M(forget)
                out = model_intergration(torch.mul(M1,outf),torch.mul(M2,outr))
                out = sigmoid(out).float()

                
                if f_alpha(epoch) != 0:
                    model = [model_rgb, model_focal, model_clstm, model_intergration]
                    out1 = correlation_samples(correlate_loader1,model)
                    out2 = correlation_samples(correlate_loader2,model)
                    out3 = correlation_samples(correlate_loader3,model)
                    
                    loss = BCE(out, y_noise) - (f_alpha(epoch)/3) * (BCE(out1, y_noise) + BCE(out1, y_noise) + BCE(out1, y_noise))
                    loss_d = F.smooth_l1_loss(dfeature_map, d, size_average = False)
                    loss_l = F.smooth_l1_loss(lfeature_map, l, size_average = False)
                    #loss_d_high = F.smooth_l1_loss(dfeature_high_map, d, size_average = False)
                    #loss_l_high = F.smooth_l1_loss(lfeature_high_map, l, size_average = False)
                    #loss_all = (loss_d + loss_l + loss_d_high + loss_l_high)/5 + loss
                    loss_all = 0.5*loss_d + 0.5*loss_l  + loss
                    
                else:
    
                    loss_d = F.smooth_l1_loss(dfeature_map, d, size_average = False)
                    loss_l = F.smooth_l1_loss(lfeature_map, l, size_average = False)
                    #loss_d_high = F.smooth_l1_loss(dfeature_high_map, d, size_average = False)
                    #loss_l_high = F.smooth_l1_loss(lfeature_high_map, l, size_average = False)
                    loss = BCE(out, y_noise)
                    #loss_all = (loss_d + loss_l + loss_d_high + loss_l_high)/5 + loss
                    loss_all = 0.5*loss_d + 0.5*loss_l  + loss
                

                optim_rgb.zero_grad()
                optim_focal.zero_grad()
                optim_clstm.zero_grad()
                optim_intergration.zero_grad()
            
                loss_all.backward()

                optim_rgb.step()
                optim_focal.step()
                optim_clstm.step()
                optim_intergration.step()
                

                if batch_idx % cfg.SYSTEM.LOG_FREQ == 0:
                    print('train loss:',loss_all.item(),'loss_crossentropy:',loss.item(),'loss_d:',loss_d.item(),'loss_l:',loss_l.item(),'epoch:',epoch,'num:',batch_idx)
                    out_save = out[0][0].cpu().data
                    imsave(os.path.join(cfg.VISUAL_ROOT, img_name[0] + '.png'), out_save, (h,w))
                
                torch.cuda.empty_cache()
            
            
        return loss_all



def train_round(cfg,iteration,loader,optimizer,model,logger,writer,Disc_Thr):
        
        epoch = 0
        optim_rgb,optim_focal,optim_clstm,optim_intergration = optimizer
        model_rgb, model_focal, model_clstm, model_intergration = model
        for epoch in range(1, cfg.SOLVER.EPOCHS+1):
            epoch_cur = epoch + iteration * cfg.SOLVER.EPOCHS
            lr = adjust_learning_rate(cfg,iteration,optim_rgb, optim_focal, optim_clstm, optim_intergration, epoch)
            _ = train_epoch(cfg,epoch_cur,loader,logger,writer,Disc_Thr,
                                    optim_rgb,optim_focal,optim_clstm,optim_intergration,
                                    model_rgb, model_focal, model_clstm, model_intergration)

            if epoch % cfg.SYSTEM.CHKPT_FREQ == 0:

                Check_point(epoch, iteration, model_rgb, model_focal, model_clstm, model_intergration)
                


def test(cfg, model, loader):
        
        device = cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu'
        model_rgb, model_focal,model_clstm, model_intergration = model
        batch_size = cfg.SOLVER.BATCH_SIZE
        num_batches = len(loader) // batch_size
        h, w = cfg.SOLVER.IMG_SIZE[0], cfg.SOLVER.IMG_SIZE[1]
        count = 0
        if not os.path.exists(cfg.MAP_ROOT):
            os.system('mkdir -p %s'%(cfg.MAP_ROOT))
            

        for batch_idx, data in enumerate(loader):
            torch.cuda.empty_cache()
            with torch.no_grad():
                count += 1
                model_rgb.eval()
                model_focal.eval()
                model_clstm.eval()
                model_intergration.eval()
    
                x = data['image'].to(device)
                fo = data['focal'].to(device)
                y = data['label'].to(device)
            
                img_name = data['img_name']

                basize, dime, height, width = fo.size()  # 2*36*256*256
                fo = fo.view(1, basize, dime, height, width).transpose(0, 1)  # 2*1*36*256*256
                fo = torch.cat(torch.chunk(fo, 12, dim=2), dim=1)  # 2*12*3*256*256
                fo = torch.cat(torch.chunk(fo, basize, dim=0), dim=1).squeeze()  # 24* 3x256x256
                
                x_90 = torch.rot90(x, 1, dims=(2,3))
                x_hori = torch.flip(x, [2])
                x_vert = torch.flip(x, [3])
                fo_90 = torch.rot90(fo, 1, dims=(2,3))
                fo_hori = torch.flip(fo, [2])
                fo_vert = torch.flip(fo, [3])
                
                f1,f2,f3,f4,f5 = model_focal(fo)
                r1,r2,r3,r4,r5 = model_rgb(x)
                outf, _, _, outr, _, _ = model_clstm(r1, r2, r3, r4, r5, f1, f2, f3, f4, f5) 
                out = model_intergration(outf,outr)
                outputs_x = sigmoid(out).float()
                del outf, outr, out, r1, r2, r3, r4, r5, f1, f2, f3, f4, f5

                
                r1, r2, r3, r4, r5 = model_rgb(x_90)
                f1, f2, f3, f4, f5 = model_focal(fo_90)
                outf, _, _, outr, _, _ = model_clstm(r1, r2, r3, r4, r5, f1, f2, f3, f4, f5) 
                out = model_intergration(outf,outr)
                outputs_x_90 = sigmoid(out).float()
                del outf, outr, out, r1, r2, r3, r4, r5, f1, f2, f3, f4, f5
                

                r1, r2, r3, r4, r5 = model_rgb(x_hori)
                f1, f2, f3, f4, f5 = model_focal(fo_hori)
                outf, _, _, outr, _, _ = model_clstm(r1, r2, r3, r4, r5, f1, f2, f3, f4, f5) 
                out = model_intergration(outf,outr)
                outputs_x_hori = sigmoid(out).float()
                del outf, outr, out, r1, r2, r3, r4, r5, f1, f2, f3, f4, f5
                
                r1, r2, r3, r4, r5 = model_rgb(x_vert)
                f1, f2, f3, f4, f5 = model_focal(fo_vert)
                outf, _, _, outr, _, _ = model_clstm(r1, r2, r3, r4, r5, f1, f2, f3, f4, f5) 
                out = model_intergration(outf,outr)
                outputs_x_vert = sigmoid(out).float()
                del outf, outr, out, r1, r2, r3, r4, r5, f1, f2, f3, f4, f5
                
                outputs_x_90 = torch.rot90(outputs_x_90, 3, dims=(2,3))
                outputs_x_hori = torch.flip(outputs_x_hori, [2])
                outputs_x_vert = torch.flip(outputs_x_vert, [3])
                
                outputs_all = outputs_x  + outputs_x_hori + outputs_x_vert + outputs_x_90
                
                outputs = outputs_all[0][0]
                outputs = outputs.cpu().data.resize_(h, w)
                imsave(os.path.join(cfg.MAP_ROOT ,img_name[0] + '.png'), outputs, cfg.SOLVER.IMG_SIZE)
                print('image ',img_name[0],'is already saved',count)
        # -------------------------- validation --------------------------- #
            torch.cuda.empty_cache()

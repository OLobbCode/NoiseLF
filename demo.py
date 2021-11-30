"""
Learning from Pixel-Level Noisy Label : A New Perspective for Light Field Saliency Detection
Submission: CVPR 2022
"""

from model import Clstm
from model import Model
from model import Collector
import torchvision
import os
from pathlib import Path
import torch
from torch import optim as optim
from torch.utils.tensorboard import SummaryWriter
from config import cfg
from dataset_loader import get_loader
from utils.basic import set_seeds
from utils.save import save_config
from utils.setup_logger import setup_logger
import torch.nn 
import torch
from trainer import *

def main():

            cfg.freeze()
            set_seeds(cfg)

            # setup logger
            logdir, chk_dir = save_config(cfg.SAVE_ROOT, cfg)
            writer = SummaryWriter(log_dir=logdir)
            logger_dir = Path(chk_dir).parent
            logger = setup_logger(cfg.SYSTEM.EXP_NAME, save_dir=logger_dir)

            # Model
            model_rgb = Model.RGBNet(cfg.SOLVER.NCLASS)
            model_focal = Model.FocalNet(cfg.SOLVER.NCLASS)
            model_clstm = Clstm.Clstm(cfg.SOLVER.NCLASS)
            model_intergration = Collector.Intergration(cfg.SOLVER.NCLASS)
            if cfg.PHASE == 'train':
                vgg19_bn = torchvision.models.vgg19_bn(pretrained=True)
                model_rgb.copy_params_from_vgg19_bn(vgg19_bn)
                model_focal.copy_params_from_vgg19_bn_focal(vgg19_bn)

            cuda = torch.cuda.is_available()
            device = torch.device(cfg.SYSTEM.DEVICE if cuda else "cpu")
            model_rgb.to(device)
            model_focal.to(device)
            model_clstm.to(device)
            model_intergration.to(device)
        




            train_loader = get_loader(cfg, cfg.DATA.TRAIN.NOISE_ROOT,'train')
            correlate_loader1 = get_loader(cfg, cfg.DATA.TRAIN.NOISE_ROOT,'cross scene')
            correlate_loader2 = get_loader(cfg, cfg.DATA.TRAIN.NOISE_ROOT,'cross scene')
            correlate_loader3 = get_loader(cfg, cfg.DATA.TRAIN.NOISE_ROOT,'cross scene')
            correlate_loaders = [train_loader, correlate_loader1, correlate_loader2, correlate_loader3]# both shuffled 

            test_loader = get_loader(cfg,'', 'test')

            num_iterations_refinement = len(cfg.SOLVER.LR)
            discretization_threshold = cfg.SOLVER.DISC
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            betas = cfg.SOLVER.BETAS


            if cfg.PHASE == 'train':
                model = [model_rgb, model_focal, model_clstm, model_intergration]
                if not os.path.exists(cfg.VISUAL_ROOT):
                    os.system('mkdir -p %s'%(cfg.VISUAL_ROOT))
                for i in range(num_iterations_refinement):
                    lr = cfg.SOLVER.LR[i]
                    optimizer_rgb = optim.Adam(model_rgb.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
                    optimizer_focal = optim.Adam(model_focal.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
                    optimizer_clstm = optim.Adam(model_clstm.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
                    optimizer_intergration = optim.Adam(model_intergration.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
                    optimizer = [optimizer_rgb, optimizer_focal, optimizer_clstm, optimizer_intergration]
                    train_round(cfg,i,correlate_loaders,optimizer,model,logger,writer,discretization_threshold)
                    discretization_threshold = 0.5

                    
            else: 
                iteration = cfg.SYSTEM.ITERATION
                model_rgb.load_state_dict(torch.load(os.path.join(cfg.SAVE_ROOT,cfg.SYSTEM.EXP_NAME,    'rgb_snapshot_' + str(iteration) + '.pth')))
                print('model_rgb ' + str(iteration)+ ' is loaded',)
                model_focal.load_state_dict(torch.load(os.path.join(cfg.SAVE_ROOT,cfg.SYSTEM.EXP_NAME, 'focal_snapshot_' + str(iteration) + '.pth')))
                print('model_focal ' + str(iteration)+ ' is loaded')
                model_clstm.load_state_dict(torch.load(os.path.join(cfg.SAVE_ROOT,cfg.SYSTEM.EXP_NAME, 'clstm_snapshot_' + str(iteration) + '.pth')))
                print('model_clstm ' + str(iteration)+ ' is loaded')
                model_intergration.load_state_dict(torch.load(os.path.join(cfg.SAVE_ROOT,cfg.SYSTEM.EXP_NAME, 'intergration_snapshot_' + str(iteration) + '.pth')))
                print('model_intergration ' + str(iteration)+ ' is loaded')

                model = [model_rgb, model_focal, model_clstm,model_intergration]
                if not os.path.exists(cfg.MAP_ROOT):
                    os.system('mkdir -p %s'%(cfg.MAP_ROOT))
                test(cfg, model, test_loader)
            

            

if __name__ == "__main__":
    main()





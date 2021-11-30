#!/usr/bin/env python
from yacs.config import CfgNode as CN

c = CN()

c.PHASE = 'test'
c.PARAM = True

c.SYSTEM = CN()
c.SOLVER = CN()

c.SYSTEM.GOON =  False
c.SYSTEM.ITERATION = 29#371750 

c.SYSTEM.EXP_NAME = 'exp_noiself'
c.SYSTEM.EXP_TYPE = 'full'
c.SYSTEM.CHKPT_FREQ = 2
c.SYSTEM.NUM_WORKERS = 0

#c.SYSTEM.DEVICE = 'cuda:3'
c.SYSTEM.NUM_GPUS = 1
c.SYSTEM.LOG_FREQ = 2
c.SYSTEM.SEED = 43

c.SOLVER.EPOCHS = 10
c.SOLVER.BATCH_SIZE = 1
c.SOLVER.NCLASS = 2
c.SOLVER.NUM_MAPS = 1
c.SOLVER.IMG_SIZE = (256,256)
c.SOLVER.LR = [1e-5, 5e-6, 1e-7]
c.SOLVER.DELTA = 0.3
c.SOLVER.ALPHA = 0.2
c.SOLVER.A = 0.04
c.SOLVER.DISC = 0.23
c.SOLVER.WEIGHT_DECAY = 0.0005
c.SOLVER.BETAS = (0.9, 0.99)

c.DATA = CN()
c.DATA.TRAIN = CN()
c.DATA.TEST = CN()
c.DATA.VAL = CN()
c.DATA.TRAIN.LIST = '../parameters/train.txt'
c.DATA.TRAIN.ROOT =  '../Light-Field_dataset/train_data' 
c.DATA.TRAIN.NOISE_ROOT = ['../train_data/noisy/RBD']  #['../train_data/noisy/DSR']
c.DATA.VAL.LIST = '../parameters/val.txt'
c.DATA.VAL.ROOT = '../Light-Field_dataset/train_data'
c.DATA.TEST.LIST = '../parameters/test.txt'
c.DATA.TEST.ROOT = '../Light-Field_dataset/test_data'



c.LOG_FILE = '../'
c.SNAPSHOT_ROOT = '../snapshot'
c.SAVE_ROOT = '../snapshot'
c.VISUAL_ROOT = '../visual7_out'
c.MAP_ROOT = '../Test/Out/'+ c.SYSTEM.EXP_NAME +'_'+ str(c.SYSTEM.ITERATION)
c.MAP_ROOT_f = '../Test/Out_f/'+ c.SYSTEM.EXP_NAME +'_'+ str(c.SYSTEM.ITERATION)
c.MAP_ROOT_r = '../Test/Out_r/'+ c.SYSTEM.EXP_NAME +'_'+ str(c.SYSTEM.ITERATION)
cfg = c


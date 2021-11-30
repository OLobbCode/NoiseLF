import os
import numpy as np
import PIL.Image as Image
import scipy.io as sio
import torch
from torch.utils import data
np.set_printoptions(threshold=np.inf)
import data_transforms as transforms
import json
from config import cfg


class MyData(data.Dataset):  # inherit
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])
    mean_focal = np.tile(mean_rgb, 12)
    std_focal = np.tile(std_rgb, 12)
    
    def __init__(self, data_root,data_list,noisy_root,transform=True):
        super(MyData, self).__init__()
        self.root = data_root
        self._transform = transform
        self.list_path = data_list
        self.noisy_path = noisy_root
     
        self.list = None

        with open(self.list_path,'r') as file:
            self.list = [x.strip() for x in file.readlines()]
        file.close()

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        img_name = self.list[index] # same as lbl_name
        
        img = load_image(os.path.join(self.root,'train_images',img_name+'.jpg'))
        lbl = load_sal_label(os.path.join(self.root,'train_masks',img_name+'.png'))
        focal = load_focal(os.path.join(self.root,'train_focal',img_name+'.mat'))
        depth = load_depthlab(os.path.join(self.root,'train_depth',img_name+'.png'))
        lab = load_depthlab(os.path.join(self.root,'train_lab',img_name+'.png'))
        noisy_lbl = []
   
        
        for noise_path in self.noisy_path:
            noisy_lbl.append(torch.Tensor(load_noisy_label(os.path.join(noise_path,img_name+'.png'))))
            
        if self._transform:
            img, focal, depth, lab = self.transform(img, focal, depth, lab)
             

        img = torch.Tensor(img)
        lbl = torch.Tensor(lbl)
        focal = torch.Tensor(focal)
        depth = torch.Tensor(depth)
        lab = torch.Tensor(lab)
        noisy_lbl = torch.stack(noisy_lbl)
        
        sample = {'image':img, 'focal':focal,'depth':depth, 'lab':lab, 'label':lbl, 'noisy_label':noisy_lbl,'img_name':img_name,'idx': index}
        return sample
        
    # Translating numpy_array into format that pytorch can use on Code.
    def transform(self, img, focal, depth, lab):#, focal, cue):
        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)  # to verify
        img = torch.from_numpy(img).float()
        
        focal = focal.astype(np.float64)/255.0
        focal -= self.mean_focal
        focal /= self.std_focal
        focal = focal.transpose(2, 0, 1)
        focal = torch.from_numpy(focal).float()

        depth = depth.astype(np.float64)/255.0
        depth = torch.from_numpy(depth).float()
        lab = lab.astype(np.float64)/255.0
        lab = torch.from_numpy(lab).float()

        return img, focal, depth, lab 
        

class MyTestData(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])
    mean_focal = np.tile(mean_rgb, 12)
    std_focal = np.tile(std_rgb, 12)

    def __init__(self, data_root,data_list,transform=True):
        super(MyTestData, self).__init__()
        self.root = data_root
        self.list_path = data_list
        self._transform = transform

        with open(self.list_path, 'r') as file:
            self.list = [x.strip() for x in file.readlines()]
        file.close()


        self.test_num = len(self.list)

    def __len__(self):
        return self.test_num

    def __getitem__(self, index):
        img_name = self.list[index % self.test_num]  # same as lbl_name
      
        img = load_image(os.path.join(self.root, 'train_images', img_name + '.jpg'))
        lbl = load_sal_label(os.path.join(self.root, 'train_masks', img_name + '.png'))
        focal = load_focal(os.path.join(self.root, 'train_focal', img_name + '.mat'))

        if self._transform:
            img,focal = self.transform(img,focal)
            
        img = torch.Tensor(img)
        lbl = torch.Tensor(lbl)
        focal = torch.Tensor(focal)

        sample = {'image':img, 'label':lbl, 'img_name':img_name,'idx': index,'focal':focal,}
        return sample

    def transform(self, img, focal):
        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        
        focal = focal.astype(np.float64)/255.0
        focal -= self.mean_focal
        focal /= self.std_focal
        focal = focal.transpose(2, 0, 1)
        focal = torch.from_numpy(focal).float()
        
      
        return img, focal


def get_loader(config, target_dirs,mode, pin=False):
    shuffle = False
    t = []
    crop_size = 256
    info = json.load(open(os.path.join('../parameters', 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t.extend([transforms.Resize_Image(crop_size),
              transforms.ToTensor(),
              normalize])
     
    if mode == 'train':
        shuffle = True
        dataset = MyData(config.DATA.TRAIN.ROOT, config.DATA.TRAIN.LIST, target_dirs,transforms.Compose(t))
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.SOLVER.BATCH_SIZE,
                                      shuffle=shuffle, num_workers=config.SYSTEM.NUM_WORKERS,
                                      pin_memory=pin, drop_last=True)
    if mode == 'cross scene':
        shuffle = True
        dataset = MyData(config.DATA.TRAIN.ROOT, config.DATA.TRAIN.LIST, target_dirs,transforms.Compose(t))
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.SOLVER.BATCH_SIZE,
                                      shuffle=shuffle, num_workers=config.SYSTEM.NUM_WORKERS,
                                      pin_memory=pin, drop_last=True)
    if mode == 'val':
        shuffle = True
        dataset = MyData(config.DATA.VAL.ROOT, config.DATA.VAL.LIST, target_dirs,transforms.Compose(t))
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.SOLVER.BATCH_SIZE,
                                      shuffle=shuffle, num_workers=config.SYSTEM.NUM_WORKERS,
                                      pin_memory=pin, drop_last=True)
    if mode == 'test':
        shuffle = False
        dataset = MyTestData(config.DATA.TEST.ROOT, config.DATA.TEST.LIST)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.SOLVER.BATCH_SIZE,
                                      shuffle=shuffle, num_workers=config.SYSTEM.NUM_WORKERS,
                                      pin_memory=pin, drop_last=True)
    return data_loader

# load image
def load_image(path, noise=False):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    img = Image.open(path)
    img = img.resize((256,256))
    img = np.array(img, dtype=np.int32)
    return img

# load noisy label
def load_sal_label(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    im = im.resize((256,256))
    label = np.array(im, dtype=np.int32)
   
    if len(label.shape) == 3:
       label = label[:,:,0]
    label = label / 255.
    label = label[np.newaxis, ...]
    label[label!=0] = 1
    return label

# load 2 cues
def load_depthlab(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    im = im.resize((256,256))
    depth = np.array(im, dtype=np.uint8)
    #label = np.array(im, dtype=np.int32)
   
    if len(depth.shape) == 3:
       depth = depth[:,:,0]
    depth = depth[np.newaxis, ...]
    return depth

# load focal
def load_focal(path):
    focal = sio.loadmat(path)
    focal = focal['img']
    focal_list = np.array_split(focal, 12, axis=2)
    focal = np.array(Image.fromarray(np.uint8(focal_list[0]),'RGB').resize((256, 256)), dtype=np.int32)
   
    count = 0
    for slice in focal_list:
        if count == 0:
            count += 1
            continue
        slice_img = Image.fromarray(np.uint8(slice),'RGB')
        slice_np = np.array(slice_img.resize((256, 256)), dtype=np.int32)
        focal = np.concatenate((focal, slice_np), axis=2)
    return focal

#omitted
def load_noisy_label(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    im = im.resize((256,256))
    label = np.array(im, dtype=np.int32)
   
    if len(label.shape) == 3:
       label = label[:,:,0]
    label = label / 255.
    label = label * 10
    label = label.astype(np.int)
    label = label.astype(np.float)
    return label

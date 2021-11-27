import numpy as np
from PIL import Image, ImageOps
import torch



#Basic Resize Transformations (used until 23.04.)
class Resize_Image(object):
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, image, gt_label, pseudo_labels, *args):
        pseudo_labels_new=[]
        for entry in pseudo_labels:
            pseudo_labels_new.append(entry.resize(self.size, Image.NEAREST))
        return image.resize(self.size, Image.CUBIC), \
               gt_label.resize(self.size, Image.NEAREST), \
               pseudo_labels_new

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, image, gt_label=None, pseudo_labels=None):
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        if gt_label is None:
            return image,
        else:
            return image, gt_label, pseudo_labels



class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic, gt_label=None, pseudo_labels=None):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic)
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':
                nchannel = 3
            else:
                nchannel = len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255)
        #handle grayscale images
        if img.shape[0]==1:
            img=torch.cat((img,img,img), dim=0)

        if gt_label is None:
            return img,
        else:
            return img, torch.LongTensor(np.array(gt_label, dtype=np.int)), [torch.LongTensor(np.array(entry, dtype=np.int)) for entry in pseudo_labels]


class Compose(object):
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

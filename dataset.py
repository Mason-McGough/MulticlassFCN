import os

from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from horsetools import list_files

    
class SegmentationDataset(Dataset):
    IMG_EXTS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    
    def _get_files(self, folder):
        if os.path.isdir(folder):
            return sorted(list_files(folder, valid_exts=self.IMG_EXTS))
        else:
            raise(RuntimeError('No folder named "{}" found.'.format(folder)))
    
    def __init__(self, root, labels, image_transforms=None, mask_transforms=None):
        self.imgs = self._get_files(os.path.join(root, 'images'))
        self.masks = self._get_files(os.path.join(root, 'masks'))
        self.labels = labels
        
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = default_loader(self.imgs[index])
        mask = default_loader(self.masks[index])
        if self.image_transforms is not None:
            img = self.image_transforms(img)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
            
        return img, mask
    
    
class LabelToOnehot(object):
    def __init__ (self, labels):
        self.labels = labels
        
    def __call__(self, img):
        img = np.array(img)
        
        if len(img.shape) > 2:
            img = img[:, :, 0]
            
        onehot = np.zeros((img.shape[0], img.shape[1], len(self.labels)))
        for i, l in enumerate(self.labels):
            onehot[:, :, i] = img == l
            
        return onehot
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def imshow(img, means=None, stds=None, title=None):
    # convert tensor back to image range [0, 1]
    img = img.numpy().transpose((1, 2, 0))
    
    if means is not None and stds is not None:
        img = np.array(stds) * img + np.array(means)
        img = np.clip(img, 0, 1)
    
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
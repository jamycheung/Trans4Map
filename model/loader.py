import os
import h5py
import json
import torch
import numpy as np
import torch.nn.functional as F
import random
from PIL import Image, ImageFilter
from torch.utils import data

import torchvision.transforms as transforms

envs_splits = json.load(open('data/envs_splits.json', 'r'))

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])


class AddSaltPepperNoise(object):

    def __init__(self, density=0, p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])
            mask = np.repeat(mask, c, axis=2)
            img[mask == 0] = 0
            img[mask == 1] = 255
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img


class Addblur(object):

    def __init__(self, p=0.5, blur="normal"):

        self.p = p
        self.blur = blur

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:

            if self.blur == "normal":
                img = img.filter(ImageFilter.BLUR)
                return img

            if self.blur == "Gaussian":
                img = img.filter(ImageFilter.GaussianBlur)
                return img

            if self.blur == "mean":
                img = img.filter(ImageFilter.BoxBlur)
                return img

        else:
            return img


normalize = transforms.Compose([
    # transforms.ToPILImage(),
    # # Addblur(p=1, blur="Gaussian"),
    # AddSaltPepperNoise(0.05, 1),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


class DatasetLoader(data.Dataset):
    def __init__(self, cfg, split='train'):
        self.split = split
        self.root = cfg['root']
        self.ego_downsample = cfg['ego_downsample']
        self.feature_type = cfg['feature_type']

        self.files = os.listdir(os.path.join(self.root, 'smnet_training_data'))

        self.files = np.array([x for x in self.files if '_'.join(x.split('_')[:2]) in envs_splits[
            '{}_envs'.format(split)]])  # using numpy format
        self.envs = np.array([x.split('.')[0] for x in self.files])  # using numpy format

        # -- load semantic map GT
        h5file = h5py.File(os.path.join(self.root, 'smnet_training_data_semmap.h5'), 'r')
        self.semmap_GT = np.array(h5file['semantic_maps'])
        h5file.close()
        self.semmap_GT_envs = json.load(open(os.path.join(self.root, 'smnet_training_data_semmap.json'), 'r'))
        self.semmap_GT_indx = {i: self.semmap_GT_envs.index(self.envs[i] + '.h5') for i in range(len(self.files))}

        assert len(self.files) > 0

        self.available_idx = np.array(list(range(len(self.files))))

    def __len__(self):
        return len(self.available_idx)

    def __getitem__(self, index):
        env_index = self.available_idx[index]

        file = self.files[env_index]
        env = self.envs[env_index]

        h5file = h5py.File(os.path.join(self.root, 'smnet_training_data', file), 'r')
        rgb = np.array(h5file['rgb'])
        depth = np.array(h5file['depth'])
        h5file.close()

        # modified
        h5file = h5py.File(os.path.join(self.root, 'smnet_training_data_maxHIndices_{}'.format(self.split), file), 'r')
        proj_indices = np.array(h5file['indices'])
        masks_outliers = np.array(h5file['masks_outliers'])
        h5file.close()


        Rgb_img = []
        Depth_img = []


        for i in range(0, 4):
            rgb_img = rgb[i]
            rgb_img = rgb_img.astype(np.float32)
            rgb_img = rgb_img / 255.0

            rgb_img = torch.FloatTensor(rgb_img).permute(2, 0, 1)
            rgb_img = normalize(rgb_img)
            rgb_img = rgb_img.unsqueeze(0)
            Rgb_img.append(rgb_img)


            depth_img = depth[i]
            depth_img = depth_img.astype(np.float32)
            depth_img = torch.FloatTensor(depth_img).unsqueeze(0)
            depth_img = depth_normalize(depth_img)
            depth_img = depth_img.unsqueeze(0)
            Depth_img.append(depth_img)

        RGB = np.concatenate(Rgb_img, axis=0)
        DEPTH = np.concatenate(Depth_img, axis=0)


        rgb = torch.from_numpy(RGB).float()
        depth = torch.from_numpy(DEPTH).float()


        proj_indices = torch.from_numpy(proj_indices).long()
        masks_outliers = torch.from_numpy(masks_outliers).bool()

        masks_inliers = ~masks_outliers

        semmap_index = self.semmap_GT_indx[env_index]
        semmap = self.semmap_GT[semmap_index]
        semmap = torch.from_numpy(semmap).long()

        return (rgb, depth, masks_inliers, proj_indices, semmap)





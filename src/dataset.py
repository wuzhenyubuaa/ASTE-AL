#!/usr/bin/python3
#coding=utf-8
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle as pkl
from tqdm import tqdm


# Classes for data augmentation, including normalization, random cropping,
# random flipping, resizing, and conversion to tensor.
class Normalize(object):
    """Normalization class."""
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std

    def __call__(self, image, mask):
        # Normalize the image using the provided mean and std values.
        image = (image - self.mean)/self.std
        # Normalize the mask to be between 0 and 1.
        mask /= 255
        return image, mask

class RandomCrop(object):
    """Randomly crop the image and mask."""
    def __call__(self, image, mask, queries=None, flag = False):
        H,W,_   = image.shape
        # Generate random offsets for cropping.
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw

        if flag:
            return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], queries[p0:p1, p2:p3]
        else:
            return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]

class RandomFlip(object):
    """Randomly flip the image and mask horizontally."""
    def __call__(self, image, mask, queries=None, flag = False):
        if np.random.randint(2)==0:
            if flag:
                return image[:, ::-1, :], mask[:, ::-1], queries[:, ::-1]
            else:
                return image[:, ::-1, :], mask[:, ::-1]
        else:
            if flag:
                return image, mask, queries
            else:
                return image, mask

class Resize(object):
    """Resize the image and mask to specified dimensions."""
    def __init__(self, H, W):
        self.H = H
        self.W = W
    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask

class ToTensor(object):
    """Convert the image and mask to PyTorch tensors."""
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        return image, mask


# Configuration class for storing parameters like mean, std, etc.
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


# Dataset class for handling data loading, preprocessing, and augmentation.
class Data(Dataset):
    """Dataset class for loading and processing images and masks."""
    def __init__(self, cfg, args, query=False):
        self.args =  args
        self.cfg        = cfg
        self.query = query
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(352, 352)
        self.totensor   = ToTensor()

        # Load the list of sample names from a text file.
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())


        # **********************active learning process***********************
        # 1. generate the initial queries
        n_pixels_per_img = args.n_pixels_per_img
        self.queries, self.n_pixels_total = None, -1
        self.dir_checkpoints = args.dir_checkpoints
        path_queries = f'{args.dir_dataset}/init_labelled_pixels.pkl'
        self.superpixel=dict()

        # Handle the initialization of queries if point-based supervision is used.
        if n_pixels_per_img != 0 and args.point_supervise == True and args.mode=='train':
            n_pixels_total = 0
            if os.path.isfile(path_queries):
                # Load pre-existing queries from a pickle file.
                self.queries = pkl.load(open(path_queries, 'rb'))
                for q in self.queries:
                    n_pixels_total += q.sum()
                self.superpixel = pkl.load(open(f'{args.dir_dataset}/superpixel_label_seeds.pkl', 'rb'))
            else:
                # Generate new queries if not already present.
                os.makedirs(f'{args.dir_checkpoints}/query_0', exist_ok=True)
                list_queries = list()

                for i in tqdm(range(len(self.samples))):
                    name = self.samples[i]
                    mask = cv2.imread(self.cfg.datapath + '/gt/' + name + '.png', 0).astype(np.float32)
                    h, w = mask.shape

                    # Flatten the mask and select a subset of pixels.
                    queries_flat = np.zeros((w*h), dtype=np.bool)
                    mask_flatten = mask.flatten()
                    index_chosen_pixels = np.random.choice(range(len(queries_flat)), n_pixels_per_img, replace=False)

                    """*************        Stage 4: (d)    **************************"""
                    if args.use_superpixel == True:
                        # If using superpixels, select pixels based on superpixel labels.
                        index_superpixels= []
                        image = cv2.imread(self.cfg.datapath+'/img/'+name+'.jpg')
                        seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], 500, 15, 3, 5, True)
                        seeds.iterate(image, 20)  # 输入图像大小必须与初始化形状相同，迭代次数为10
                        mask_seeds = seeds.getLabelContourMask()
                        label_seeds = seeds.getLabels()
                        number_seeds = seeds.getNumberOfSuperpixels()
                        label_flatten= label_seeds.flatten()

                        if number_seeds < 256 :
                            self.superpixel[name] = label_flatten.astype(np.uint8)
                        else:
                            self.superpixel[name]=label_flatten.astype(np.int16)

                        for id in index_chosen_pixels:
                            index_sup = np.where(label_flatten==label_flatten[id])
                            index_superpixels.append(index_sup[0])
                            queries_flat[index_sup[0]] = True
                    else:
                        # If not using superpixels, select random pixels.
                        queries_flat[index_chosen_pixels] = True

                    # Reshape the query mask to match the image dimensions.
                    queries_flat.resize((h, w))
                    queries = queries_flat
                    list_queries.append(queries)
                    n_pixels_total += queries.sum()

                self.queries = list_queries

                # Save the generated queries and superpixel labels to pickle files.
                pkl.dump(list_queries, open(f'{path_queries}', 'wb'))
                pkl.dump(self.superpixel, open(f'{args.dir_dataset}/superpixel_label_seeds.pkl', 'wb'))
                pkl.dump(self.queries, open(f'{self.dir_checkpoints}/query_0/label.pkl',  'wb'))

            self.n_pixels_total = n_pixels_total


    def update_query(self, queries, nth_query=None, list_names=None):
        """Update the query list with new information."""
        assert list_names == self.samples, f'the order of queries is unmatched!'
        assert len(queries) == len(self.queries), f"{queries.shape}, {len(self.queries)}"
        list_queries = list()
        n_pixels_total = 0

        for q, m in zip(queries, self.queries):
            new_m = np.logical_or(q, m)
            list_queries.append(new_m)
            n_pixels_total += new_m.sum()
        self.queries, self.n_pixels_total = list_queries, n_pixels_total

        if isinstance(nth_query, int):
            os.makedirs(f"{self.dir_checkpoints}/query_{nth_query}", exist_ok=True)
            pkl.dump(self.queries, open(f"{self.dir_checkpoints}/query_{nth_query}/label.pkl", 'wb'))

        print("# ***********labelled pixels is updated*************")

    def __getitem__(self, idx):
        """Get a single data sample."""
        name  = self.samples[idx]

        if self.query:  # when using dataload_query
            # If querying, return additional information like image gray scale and horizontal flip.
            image = cv2.imread(self.cfg.datapath + '/img/' + name + '.jpg')[:, :, ::-1].astype(np.float32) # color image
            mask = cv2.imread(self.cfg.datapath + '/gt/' + name + '.png', 0).astype(np.float32)
            queries = self.queries[idx]
            image, mask = self.normalize(image, mask)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_hflip = image[:,::-1,:]

            return image, mask, queries, name

        image = cv2.imread(self.cfg.datapath+'/image/'+name+'.jpg')[:,:,::-1].astype(np.float32)
        mask  = cv2.imread(self.cfg.datapath+'/mask/' +name+'.png', 0).astype(np.float32)
        shape = mask.shape

        if self.args.point_supervise==True:   # point supervise

            if self.cfg.mode=='train':
                queries = self.queries[idx]
                image, mask = self.normalize(image, mask)
                image, mask, queries = self.randomcrop(image, mask, queries=queries, flag=True)
                image, mask, queries = self.randomflip(image, mask, queries=queries, flag=True)
                return image, mask, queries, name
            else:
                image, mask = self.normalize(image, mask)
                image, mask = self.resize(image, mask)
                image, mask = self.totensor(image, mask)
                return image, mask, shape, name
        else:# full-supervise
            if self.cfg.mode=='train':
                image, mask = self.normalize(image, mask)
                image, mask,  = self.randomcrop(image, mask, )
                image, mask,  = self.randomflip(image, mask, )
                return image, mask,
            else:
                image, mask = self.normalize(image, mask)
                image, mask = self.resize(image, mask)
                image, mask = self.totensor(image, mask)
                return image, mask, shape, name

    def collate(self, batch):

        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        if self.args.point_supervise == True:
            image, mask, queries, name = [list(item) for item in zip(*batch)]
            for i in range(len(batch)):
                image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
                mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
                tmp = np.where(queries[i]==True, 1, 0)
                queries[i] = cv2.resize(tmp,  dsize=(size, size), interpolation=cv2.INTER_NEAREST) # should be INTER_NEAREST
                queries[i] = np.asarray(queries[i], dtype=np.bool)

            image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
            mask  = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
            queries = torch.from_numpy(np.stack(queries, axis=0)).unsqueeze(1)
            return image, mask, queries, name
        else:
            image, mask = [list(item) for item in zip(*batch)]
            for i in range(len(batch)):
                image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
                mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
            mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)

            return image, mask

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)



if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    cfg  = Config(mode='train', datapath='../data/DUTS')
    data = Data(cfg)
    for i in range(1000):
        image, mask = data[i]
        image       = image*cfg.std + cfg.mean
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask)
        input()

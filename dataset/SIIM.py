import torch
import torch.nn 
from torch.utils.data import Dataset

import numpy as np 
import cv2 
from PIL import Image
import glob
import sys 
import os
import pandas as pd
sys.path.append('.')
import pydicom

from data import mask_functions
from sklearn import model_selection
from dataset import augmentation

seed = 1234



class SIIM_ACR(Dataset):

    def __init__(self,mode='train',k_fold = 5, split = 1, preprocess=None, augmentation=None, use_total= False, is_tta=False):
        
        if split > k_fold:
            raise ValueError('Please use split less than k_fold')

        self._mode = mode
        self._preprocess = preprocess
        self._augmentation = augmentation
        self._split = split
        self._is_tta = is_tta
        self.get_file()

        kf = model_selection.KFold(n_splits=k_fold,shuffle=True,random_state=seed)

        self.train_fold = {}
        self.valid_fold = {}
        self.weight_train_fold = {}
        self.weight_test_fold = {}
        for idx, (train_index, test_index) in enumerate(kf.split(self.train_path)):
            x_train, x_test = self.train_path[train_index], self.train_path[test_index]
            weight_label_train, weight_label_test = self.weight_label[train_index], self.weight_label[test_index]
            self.train_fold[idx] = x_train
            self.valid_fold[idx] = x_test

            self.weight_train_fold[idx] = weight_label_train
            self.weight_test_fold[idx] = weight_label_test

        
        if mode == 'train':
            self.data = self.train_fold[split]
            self.weight = self.weight_train_fold[split]
        else:
            self.data = self.valid_fold[split]
            self.weight = self.weight_test_fold[split]

        if use_total == True and mode == 'train':
            self.data = self.train_path
            self.weight = self.weight_label

    def get_file(self):
        
        # non_diagnostic = np.genfromtxt("/4T/Public/zhujian/siim_acr/dataset/no_diagnostic.txt",'U')

        self.train_path = glob.glob(os.path.join("/4T/Public/zhujian/siim_acr/data/train/",'*'))
        self.train_path.sort()
        self.train_path = np.array(self.train_path)

        self.weight_label = np.load("/4T/Public/zhujian/siim_acr/data/black_list.npy")
        self.weight_label = np.array(self.weight_label)
        # self.test_path = glob.glob(os.path.join("/4T/Public/zhujian/siim_acr/data/dicom-images-test/",'*','*','*'))
        # self.test_path = np.array(self.test_path)

        self.train_label = pd.read_csv("/4T/Public/zhujian/siim_acr/data/train-rle.csv",header=None,index_col=0)


    def __getitem__(self,idx):
        
        # dcm_path = self.data[idx]

        # dataset = pydicom.dcmread(dcm_path)
        # height, width = int(dataset.Rows), int(dataset.Columns)
        # image = dataset.pixel_array
        # mask = np.zeros((height, width))

        # try:
        #     label = self.train_label.loc[dcm_path.split('/')[-1][:-4], 1]
        #     if type(label) != str:
        #         for i in range(label.shape[0]):
        #             l = label.iloc[i]
        #             m = mask_functions.rle2mask(l, 1024, 1024).T
        #             mask += m

        #     elif label != ' -1':
        #         mask = mask_functions.rle2mask(label, 1024, 1024).T
        # except:
        #     pass

        image_path = os.path.join(self.data[idx],'image.jpg')
        label_path = os.path.join(self.data[idx],'label.jpg')
        image = cv2.imread(image_path)
        mask = cv2.imread(label_path,0)

        data = {'image':image,'mask':mask}

        # print(len(np.unique(mask)), self.weight[idx])
        assert int(len(np.unique(mask)) != 1) == self.weight[idx]

        if self._augmentation is not None:
            aug_data = self._augmentation(**data)
            image, mask = aug_data['image'], aug_data['mask']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        if self._is_tta:
            image = cv2.flip(image,1)
            mask = cv2.flip(mask,1)

        if self._preprocess is not None:
            image = self._preprocess(image)
        else:
            image = image / 127.5 - 1

        mask = mask / 255.0
        
        uni = np.unique(mask)
        if len(uni) == 1:
            is_black = 0
        else:
            is_black = 1

        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask >= 1.0)

        image = image.permute(2,0,1)
        mask = mask.unsqueeze(0)
        
        black = torch.FloatTensor([is_black])

        label = {'mask':mask,'black':black}

        return image, label

    def __len__(self):

        return len(self.data)
        

if __name__ == "__main__":
    a = SIIM_ACR()

    train_dataloader = torch.utils.data.DataLoader(
        a,
        batch_size=8, shuffle=True, num_workers=4, pin_memory=True
    )
    import time
    e = time.time()
    for i, (image,mask) in enumerate(train_dataloader):
        if i == 1:
            break
        print(image.shape, label['mask'].shape)
        e = time.time()
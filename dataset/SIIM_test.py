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


class dcm(object):

    def __init__(self, data):
        self._data = data

    @property
    def imageid(self):

        return self._data['imageid']



class SIIM_ACR(Dataset):

    def __init__(self, mode='test', preprocess=None, augmentation=None):


        self._mode = mode
        self._preprocess = preprocess
        self._aug = augmentation
        self.get_file()


        self.data = self.test_path

    def get_file(self):

        self.test_path = glob.glob(os.path.join('/4T/Public/zhujian/siim_acr/data/dicom-images-test/', '*', '*', '*'))
        self.test_path = np.array(self.test_path)


    def __getitem__(self, idx):

        dcm_path = self.data[idx]

        dataset = pydicom.dcmread(dcm_path)
        height, width = int(dataset.Rows), int(dataset.Columns)
        image = dataset.pixel_array

        data = {'image': image}

        aug_data = self._aug(**data)
        image = aug_data['image']
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self._preprocess is not None:
            image = self._preprocess(image)
        else:
            image = image / 127.5 - 1

        image = torch.FloatTensor(image)

        image = image.permute(2, 0, 1)

        return image, dcm_path.split('/')[-1][:-4]

    def __len__(self):

        return len(self.data)


if __name__ == "__main__":
    a = SIIM_ACR()

    train_dataloader = torch.utils.data.DataLoader(
        a,
        batch_size=1, shuffle=True, num_workers=4, pin_memory=False
    )
    import time

    e = time.time()
    for i, (image, imageid) in enumerate(train_dataloader):
        print(imageid,time.time() - e)
        e = time.time()
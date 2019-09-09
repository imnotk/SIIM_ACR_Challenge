from albumentations import HorizontalFlip, ShiftScaleRotate, RandomContrast, RandomBrightness, Compose, RandomCrop, RandomGamma, OneOf
from albumentations import Resize, CenterCrop, RandomScale, RandomCrop, RandomCropNearBBox, RandomSizedBBoxSafeCrop, ElasticTransform, GridDistortion, OpticalDistortion, RandomSizedCrop
from albumentations import Rotate, RandomSizedCrop, RandomBrightnessContrast, GaussNoise
import cv2

def get_augmentations(augmentation, p, image_size = 256):
    if augmentation == 'train':
        augmentations = Compose([
            # RandomScale(scale_limit=0.125),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.5),
            RandomGamma(p=0.3),
            ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=15, p=0.6),
            # ShiftScaleRotate(rotate_limit=20, p=0.6),
            Resize(image_size,image_size)
            
        ], p=p)
    elif augmentation == 'valid':
        augmentations = Compose([
            Resize(image_size, image_size)
        ], p=p)
    else:
        raise ValueError("Unknown Augmentations")

    return augmentations
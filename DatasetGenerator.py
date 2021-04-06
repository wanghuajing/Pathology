from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import random
from matplotlib import pyplot as plt
import cv2
import torch
import pydicom
import numpy as np


def get_center(mask):
    mass_x, mass_y = np.where(mask >= 255)
    # mass_x and mass_y are the list of x indices and y indices of mass pixels

    cent_x = int(np.average(mass_x))
    cent_y = int(np.average(mass_y))
    return [cent_x, cent_y]


def get_mask(image, center):
    left = center[1] - 256
    right = center[1] + 256
    top = center[0] - 256
    bottom = center[0] + 256
    if left < 0:
        right = right - left
        left = 0
    if right > image.shape[0] - 1:
        left = left - (right - image.shape[0] + 1)
        right = image.shape[0] - 1
    if top < 0:
        bottom = bottom - top
        top = 0
    if bottom > image.shape[1] - 1:
        top = top - (bottom - image.shape[1] + 1)
        bottom = image.shape[1] - 1
    return image[top:bottom, left:right]


def make_pathology(pathology):
    return 1.0 if pathology == "MALIGNANT" else 0.0


class DatasetGenerator(Dataset):

    # --------------------------------------------------------------------------------

    def __init__(self, df, path):
        self.df = df
        self.path = path

    # --------------------------------------------------------------------------------

    def __getitem__(self, index):
        dcm = pydicom.read_file(self.path + self.df.iloc[index, 1])
        mask = dcm.pixel_array
        dcm = pydicom.read_file(self.path + self.df.iloc[index, 0])
        image = dcm.pixel_array
        center = get_center(mask)
        image = get_mask(image, center)
        mask = get_mask(mask, center).astype(np.uint16)
        mask[mask == 255] = 2 ** 16 - 1
        image = cv2.merge([image, image, mask])
        labels = make_pathology(self.df.iloc[index, 2])

        return image, labels

    # --------------------------------------------------------------------------------

    def __len__(self):
        return self.df.shape[0]

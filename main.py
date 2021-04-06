from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import random
from matplotlib import pyplot as plt
import cv2
import torch
import pydicom
import numpy as np
from DatasetGenerator import DatasetGenerator
import config
import os
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    dataset = DatasetGenerator(config.table, config.path)
    # for i, (image, label) in enumerate(dataset):
    #     print(label)
    resnet18 = models.resnet18(pretrained=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

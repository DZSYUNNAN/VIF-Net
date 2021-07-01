import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import PIL
from PIL import Image
#from flowlib import read, read_weights_file
from skimage import io, transform
from PIL import Image
import numpy as np
import re

device = 'cuda'


def toString(num):
    string = str(num)
    while (len(string) < 4):
        string = "0" + string
    return string


class ImageDataset(Dataset):

    def __init__(self, infrared_dataroot, visible_dataroot, image_size):
        """
        looking at the "clean" subfolder for images, might change to "final" later
        root_dir -> path to the location where the "training" folder is kept inside the MPI folder
        """
        self.infrared_dataroot = infrared_dataroot
        self.visible_dataroot=visible_dataroot
        self.image_size = image_size
        self.total_image = []

        for item in os.listdir(os.path.join(self.infrared_dataroot)):

            print('It is now processing {} !'.format(item))
            source_img_dir = os.path.join(self.infrared_dataroot, item)
            source_image_list = os.listdir(os.path.join(source_img_dir))
            source_image_list.sort(key=lambda x: str(re.split('\.|\_', x)[1]))
            target_image_dir = os.path.join(self.visible_dataroot, item)
            target_image_list = os.listdir(os.path.join(target_image_dir))
            target_image_list.sort(key=lambda x: str(re.split('\.|\_', x)[1]))
            #print('target_image_list',target_image_list)
            #tmp_image = (target_image_list, source_image_list)
            tmp_len = len(source_image_list) - 1
            for i in range(tmp_len):
                target_img = os.path.join(target_image_dir, target_image_list[i])
                source_img = os.path.join(source_img_dir, source_image_list[i])
                tmp_image = (target_img, source_img)
                self.total_image.append(tmp_image)
        self.lens = len(self.total_image)
        self.transform = transforms.Compose([
            transforms.TenCrop(self.image_size), #crop images
            transforms.ToTensor()])

    def __len__(self):
        return self.lens

    def __getitem__(self, i):
        """
        idx must be between 0 to len-1
        assuming flow[0] contains flow in x direction and flow[1] contains flow in y
        """
        image_path1 = self.total_image[i][0]
        image_path2 = self.total_image[i][1]
        # print('image_path is ',image_path1)
        # print('backward_flow_path ',backward_flow_path)
        # print('consistency_path ',consistency_path)
        img1 = Image.open(image_path1).convert('L')
        img2 = Image.open(image_path2).convert('L')
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        # mask is numpy and shape is HXWX3
        # img_flow = Image.fromarray(np.uint8(flow))
        # this is for transpose.
        # [3,400,640] [3,400,640] [3,400,640],[ 2, 400, 640]
        return (img1, img2)


if __name__ == "__main__":
    image = ImageDataset()
    print('data lens', len(image))
    dataloader = torch.utils.data.DataLoader(image, batch_size=1)
    for index, item in enumerate(dataloader):
        print(index)
        print(item[0].shape)
        print(item[1].shape)





import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from models import fusion_model
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import ImageFolder
from input_data import ImageDataset
import time
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument("--infrared_dataroot", default="E:/shujuji/IR/", type=str)
parser.add_argument("--visible_dataroot", default="E:/shujuji/VIS/", type=str)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--output_root", default="./outputs/", type=str)
parser.add_argument("--image_size", type=int, default=[128, 128])
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")

if __name__ == "__main__":
    opt = parser.parse_args()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    device = torch.device('cuda:0')
    if not os.path.exists(opt.output_root):
        os.makedirs(opt.output_root)
    net = fusion_model.FusionNet().to(device)
    net.load_state_dict(torch.load("./checkpoints/fusion_last_10.pth"))
    net.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    with torch.no_grad():
        for i in range(20):
            start = time.time()
            index = i + 1
            infrared = Image.open(opt.infrared_dataroot + 'IR' + str(index) + '.png').convert('L')
            infrared = transform(infrared).unsqueeze(0)
            visible = Image.open(opt.visible_dataroot + 'VIS'+ str(index) + '.png').convert('L')
            visible = transform(visible).unsqueeze(0)
            infrared = infrared.to(device)
            visible = visible.to(device)
            fused_img = net(infrared,visible)
            save_image(fused_img.cpu(), os.path.join(opt.output_root, str(index) + ".jpg"))
            end = time.time()
            print('consume time:',end-start)
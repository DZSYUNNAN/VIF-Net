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
from pytorch_ssim import ssim, tv_loss
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)
torch.set_num_threads(6)

parser = argparse.ArgumentParser()
parser.add_argument("--infrared_dataroot", default="E:/shujuji/TNO//IR/", type=str)
parser.add_argument("--visible_dataroot", default="E:/shujuji/TNO/VIS/", type=str)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--image_size", type=int, default=[128, 128])
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")


if __name__ == "__main__":
    opt = parser.parse_args()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    device = torch.device('cuda:0')
    net = fusion_model.FusionNet().to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad,net.parameters()),lr=opt.lr)
    train_datasets = ImageDataset(opt.infrared_dataroot, opt.visible_dataroot, opt.image_size)
    lens = len(train_datasets)
    log_file = './log_dir'
    dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=opt.batch_size,shuffle=False)
    runloss = 0.
    total_params = sum(p.numel() for p in net.parameters())
    print('total parameters:', total_params)
    weight = 10
    for epoch in range(opt.epoch):
        #if epoch % 5==1:
         #   opt.lr=0.1*opt.lr
        for index, data in enumerate(dataloader):
            bs, nc, c, h, w = data[0].size()
            infrared = data[0].view(-1, c, h, w).to(device)
            visible = data[1].view(-1, c, h, w).to(device)
            fused_img = net(infrared, visible)
            LOSS_SSIM = 1-ssim(fused_img, infrared, visible)
            LOSS_TV = tv_loss(fused_img-visible)
            loss = weight*LOSS_SSIM + LOSS_TV
            runloss += loss.item()
            print('epoch [{}/{}], images [{}/{}], SSIM loss is {:.5}, TV loss is {:.5}, total loss is  {:.5}, lr: {}'.
                  format(epoch + 1, opt.epoch, (index + 1) * data[0].shape[0], lens, LOSS_SSIM.item(),LOSS_TV.item(), loss.item(), opt.lr))
            runloss = 0.
            optim.zero_grad()
            loss.backward()
            optim.step()
    torch.save(net.state_dict(), './checkpoints/fusion_last_10.pth'.format(opt.lr, log_file[2:]))
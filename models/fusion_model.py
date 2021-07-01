import torch
import torch.nn as nn
import torch.nn.parallel
class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1),
                                     #nn.BatchNorm2d(1),
                                     nn.ReLU(),
                                     )
        self.conv1_2 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1),
                                     #nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     )
        self.conv1_3 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1),
                                     #nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     )
        self.conv1_4 = nn.Sequential(nn.Conv2d(48, 16, 3, 1, 1),
                                     #nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     )#output 64*64
        self.conv2_1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1),
                                     #nn.BatchNorm2d(1),
                                     nn.ReLU(),
                                     )
        self.conv2_2 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1),
                                     #nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     )
        self.conv2_3 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1),
                                     #nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     )
        self.conv2_4 = nn.Sequential(nn.Conv2d(48, 16, 3, 1, 1),
                                     #nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     )
        self.conv3_1 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1),
                                     #nn.BatchNorm2d(128),
                                     nn.ReLU()
                                     )
        self.conv3_2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                     #nn.BatchNorm2d(64),
                                     nn.ReLU()
                                     )
        self.conv3_3 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3_4 = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, infrared, visible):
        conv1_1 = self.conv1_1(infrared)
        conv1_2 = self.conv1_2(conv1_1)
        conv1_3 = self.conv1_3(torch.cat((conv1_2, conv1_1), dim=1))
        conv1_4 = self.conv1_4(torch.cat((conv1_3, torch.cat((conv1_2, conv1_1), dim=1)), dim=1))
        conv2_1 = self.conv2_1(visible)
        conv2_2 = self.conv2_2(conv2_1)
        conv2_3 = self.conv2_3(torch.cat((conv2_2, conv2_1), dim=1))
        conv2_4 = self.conv2_4(torch.cat((conv2_3, torch.cat((conv2_2, conv2_1), dim=1)), dim=1))
        concate = torch.cat((torch.cat((conv1_4, torch.cat((conv1_3, torch.cat((conv1_2, conv1_1), dim=1)), dim=1)), dim=1), torch.cat((torch.cat((torch.cat((conv2_1, conv2_2), dim=1), conv2_3), dim=1), conv2_4), dim=1)), dim=1)
        conv3_1 = self.conv3_1(concate)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        conv3_4 = self.conv3_4(conv3_3)
        return conv3_4

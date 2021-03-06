import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import time
import os
import numpy as np

# Codes from https://deep-learning-study.tistory.com/534
# 추가된 부분: ResNet 처음에 input 받아서 64채널로 변환하는 conv layer에서
#              input channel number를 수정하도록 변경


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10, n_cv=0, init_weights=True):
        super().__init__()

        self.in_channels=64

        # for gabor
        self.first_layer_input_size = self.in_channels
        self.first_layer_output_size = 64
        self.first_layer_stride = 2
        self.first_layer_kernel_size = 7
        self.first_layer_padding = 3

        # Traditional Computer Vision Technique를 이용한 preprocess로
        # input channel이 증가한 경우 n_cv를 통해 이를 network에 반영
        # First convolution layer의 input channel 수를 변경가능
        self.conv1 = nn.Sequential(
            nn.Conv2d((3+n_cv), 64, 
                        kernel_size=self.first_layer_kernel_size, 
                        stride=self.first_layer_stride, 
                        padding=self.first_layer_padding, 
                        bias=False
                    ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, self.first_layer_output_size, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        
        # dirname = time.time()
        # print(output.shape)
        # os.mkdir(f"samples/{dirname}")
        # x_ = torch.mean(x, axis=1)[0].squeeze().detach().cpu().numpy()*255
        # output_img = Image.fromarray(x_).convert("L")
        # output_img.save(f"samples/{dirname}/first_layer_x.jpg")
        # for channel in range(output.shape[1]):
        #     ou = output[0][channel].detach().cpu().numpy()
        #     output_img = Image.fromarray(ou*255/np.max(ou)).convert("L")
        #     output_img.save(f"samples/{dirname}/first_layer_{channel}.jpg")

        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet18(n_class, n_cv):
    return ResNet(BasicBlock, [2,2,2,2], n_class, n_cv)

def resnet34(n_class, n_cv):
    return ResNet(BasicBlock, [3,4,6,3], n_class, n_cv)

def resnet50(n_class, n_cv):
    return ResNet(BottleNeck, [3,4,6,3], n_class, n_cv)

def resnet101(n_class, n_cv):
    return ResNet(BottleNeck, [3,4,23,3], n_class, n_cv)

def resnet152(n_class, n_cv):
    return ResNet(BottleNeck, [3,8,36,3], n_class, n_cv)
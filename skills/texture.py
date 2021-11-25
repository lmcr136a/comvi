import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
import math
import time
import os 


"""
    gabor filter explanation:
    http://www.cs.rug.nl/~imaging/simplecell.html
    ** 채널 3개 증가
    Args:
        kszie : tuple. kernel size
        sigma : double sigma
        theta : gabor filter 방향(radian)
        gamma : gaussian의 elipcity결정    
        psi : phase offset
        ktype : CV_32F or CV_64F
"""
class GABOR(object):
    def __init__(self,ksize1=50,ksize2 = 50,sigma=5,theta=0,lambd=10,gamma=1,psi=0,ktype=cv.CV_64F):
        self.ksize = (ksize1,ksize2)
        self.sigma = sigma
        self.theta = theta
        self.lambd = lambd
        self.gamma = gamma        
        self.psi = psi
        self.ktype = ktype

    def __call__(self,torch_img):
        img = torch_img[0:3,:,:].detach().numpy().astype(np.uint8)
        kernel = cv.getGaborKernel(self.ksize, self.sigma, self.theta, self.lambd,self.gamma, self.psi, self.ktype)
        tex = cv.filter2D(img, -1, kernel)
        tex = torch.Tensor(tex)
        tmp = torch.cat((torch_img, tex),dim=0)
        return tmp


class GaborLayerLearnable(nn.Module):
    def __init__(self, in_channels, stride, padding, out_channels, 
        kernel_size, bias1=False, bias2=False, relu=True, 
        use_alphas=True, use_translation=False):
        super(GaborLayerLearnable, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernels = out_channels
        self.total_kernels = self.kernels
        self.responses = self.kernels
        self.kernel_size = kernel_size
        self.relu = relu
        self.use_alphas = use_alphas
        self.use_translation = use_translation
        # All parameters have the same size as the number of families and input channels
        # Replicate parameters so that broadcasting is possible
        self.Lambdas = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
            dim=1).unsqueeze(dim=2))
        self.psis = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
            dim=1).unsqueeze(dim=2))
        self.sigmas = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
            dim=1).unsqueeze(dim=2))
        self.gammas_y = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
            dim=1).unsqueeze(dim=2))
        
        if self.use_translation:
            self.trans_x = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
                dim=1).unsqueeze(dim=2))
            self.trans_y = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
                dim=1).unsqueeze(dim=2))

        if self.use_alphas:
            self.alphas = nn.Parameter(torch.randn(self.responses).unsqueeze(
                dim=1).unsqueeze(dim=2))
        # Bias parameters start in zeros
        self.bias = nn.Parameter(torch.zeros(self.responses)) if bias1 else None
        # # # # # # # # # # # # # # The meshgrids
        # The orientations (NOT learnt!)
        self.thetas = nn.Parameter(torch.randn(self.total_kernels).unsqueeze(
            dim=1).unsqueeze(dim=2))
        # The original meshgrid
        # Bounding box
        xmin, xmax = -1, 1
        ymin, ymax = -1, 1
        x_space = torch.linspace(xmin, xmax, kernel_size)
        y_space = torch.linspace(ymin, ymax, kernel_size)
        (y, x) = torch.meshgrid(y_space, x_space)
        # Unsqueeze for all orientations
        x, y = x.unsqueeze(dim=0), y.unsqueeze(dim=0)
        # Add as parameters
        self.x = nn.Parameter(x, requires_grad=False)
        self.y = nn.Parameter(y, requires_grad=False)
        # Conv1x1 channels
        self.channels1x1 = self.responses * self.in_channels
        self.conv1x1 = nn.Conv2d(
            in_channels=self.channels1x1, out_channels=out_channels, 
            kernel_size=1, bias=bias2)
        self.gabor_kernels = nn.Parameter(
            self.generate_gabor_kernels(), requires_grad=False)

    def forward(self, x):
        # Generate the Gabor kernels
        if self.training:
            self.gabor_kernels = nn.Parameter(
                self.generate_gabor_kernels(), requires_grad=False)
        # kernels are of shape 
        # [self.kernels*self.orientations + self.extra_kernels, 1, self.kernel_size, self.kernel_size]
        # Reshape the input: x is of size
        # [batch_size, in_channels, H, W]
        # and we need to merge the batch size and the input channels for the
        # depthwise convolution (and include a '1' channel for the convolution)
        b, c, H, W = x.size()
        if c > 1:
            x = torch.mean(x, axis=1)
            x = x.unsqueeze(dim=1)
        # Perform convolution
        out = nn.functional.conv2d(input=x, weight=self.gabor_kernels, bias=self.bias, 
            stride=self.stride, padding=self.padding)
        if self.relu:
            out = torch.relu(out)
        return out

    '''
    Inspired by
    https://en.wikipedia.org/wiki/Gabor_filter
    '''
    def generate_gabor_kernels(self):
        sines = torch.sin(self.thetas)
        cosines = torch.cos(self.thetas)
        x = self.x * cosines - self.y * sines
        y = self.x * sines + self.y * cosines
        # Precompute some squared terms
        # ORIENTED KERNELS
        # Compute Gaussian term
        # gaussian_term = torch.exp(-.5 * ( (gamma_x**2 * x_t**2 + gamma_y**2 * y_t**2)/ sigma**2 ))
        ori_y_term = (self.gammas_y * y)**2
        exponent_ori = (x**2 + ori_y_term) * self.sigmas**2
        gaussian_term_ori = torch.exp(-exponent_ori)
        # Compute Cosine term
        # cosine_term = torch.cos(2 * np.pi * x_t / Lambda + psi)
        cosine_term_ori = torch.cos(x * self.Lambdas + self.psis)
        # 'ori_gb' has shape [self.kernels, self.orientations, kernel_size, kernel_size]
        ori_gb = gaussian_term_ori * cosine_term_ori
        if self.use_alphas:
            ori_gb = self.alphas * ori_gb


        # output = ori_gb*255
        # print(output.shape)
        # dirname = "gabor_"+str(time.time())
        # os.mkdir(f"samples/{dirname}")
        # for channel in range(output.shape[0]):
        #     output_img = Image.fromarray(output[channel].detach().cpu().numpy()).convert("L").resize((200,200))
        #     output_img.save(f"samples/{dirname}/first_layer_{channel}.jpg")

        return ori_gb.unsqueeze(dim=1)



if __name__ == '__main__':
    gabor = GABOR((21,21), 5, 1, 10, 1, 0, cv.CV_64F)
    sample_image = Image.open("data_for_test/testimg.jpg")
    sample_image = torch.Tensor(np.array(sample_image))
    shape = sample_image.shape
    sample_image = sample_image.reshape(shape[2],shape[0],shape[1])
    tex = gabor(sample_image)
    tex = tex.reshape(tex.shape[1],tex.shape[2],tex.shape[0])

    plt.figure()
    plt.imshow(tex[:,:,3:])
    plt.show()
import numpy as np
import cv2 as cv
import torch
from PIL import Image
from matplotlib import pyplot as plt
import math
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

# if __name__ == '__main__':
#     gabor = GABOR((21,21), 5, 1, 10, 1, 0, cv.CV_64F)
#     sample_image = Image.open("data_for_test/testimg.jpg")
#     sample_image = torch.Tensor(np.array(sample_image))
#     shape = sample_image.shape
#     sample_image = sample_image.reshape(shape[2],shape[0],shape[1])
#     tex = gabor(sample_image)
#     tex = tex.reshape(tex.shape[1],tex.shape[2],tex.shape[0])

#     plt.figure()
#     plt.imshow(tex[:,:,3:])
#     plt.show()
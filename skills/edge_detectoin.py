import numpy as np
import cv2 as cv
import torch
from PIL import Image
from matplotlib import pyplot as plt

class EDGE(object):
    """
    Detect edge and spit out in tensor form.
    **채널 1개증가
    Args:
        img (torch array) : 엣지 디텍트 할 이미지.
        lthr(int) : low threshold
        hthr(int) : high threshold
    """
    def __init__(self,lthr=150,hthr=200):
        self.lthr = lthr
        self.hthr = hthr
    
    def __call__(self,torch_img):
        img = torch_img[0:3,:,:].detach().numpy().astype(np.uint8)
        edge = cv.Canny(img.reshape(img.shape[1],img.shape[2],img.shape[0]),self.lthr,self.hthr)
        edge = torch.Tensor(edge)
        edge = torch.unsqueeze(edge,0)
        return torch.cat((torch_img, edge),dim=0)

if __name__ == '__main__':
    edge_detector = EDGE(150,200)
    sample_image = Image.open("data_for_test/testimg.jpg")
    sample_image = torch.Tensor(np.array(sample_image))
    shape = sample_image.shape
    sample_image = sample_image.reshape(shape[2],shape[0],shape[1])
    edge = edge_detector(sample_image)

    plt.figure()
    plt.imshow(edge[3,:,:])
    plt.show()
import math
import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
"""
Saliency map
"""

class SaliencyMap(object):
    def __init__(self, mask=True):
        self.mask = mask

    def __call__(self, torch_img, kernel_size=(5,5)):
        img = torch_img.detach().numpy().astype(np.uint8)
        if img.shape[0] == 3:
            axis = 0
            gray = np.mean(img, axis=0).astype(np.uint8)
        else:
            axis = 2
            gray = np.mean(img, axis=2).astype(np.uint8)
        resize_img = img
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (suc, s_map) = saliency.computeSaliency(resize_img)
        if not suc:
            return None
        s_map = (s_map*255).astype("uint8")
        
        if self.mask:
            thresh_map = cv2.threshold(s_map, 0, 255, cv2.THRESH_BINARY | 8)[1]
            thresh_map_3c = np.expand_dims(thresh_map, axis=2)
            thresh_map_3c = np.concatenate((thresh_map_3c,thresh_map_3c,thresh_map_3c), axis=2)
            saliency_img = np.array(img)/255*thresh_map_3c
        else:
            s_map_3c = np.expand_dims(s_map, axis=2)
            s_map_3c = np.concatenate((s_map_3c,s_map_3c,s_map_3c), axis=2)
            saliency_img = np.array(img)/255*s_map_3c

        return torch.cat((torch_img, torch.Tensor(saliency_img)), axis=axis)


if __name__ == '__main__':
    sample_image = Image.open("data/cat_dog/test/dog/dog.12400.jpg")
    sample_image = torch.Tensor(np.array(sample_image))
    sm = SaliencyMap(mask=True)
    siftimg = sm(sample_image)
    print(siftimg.shape)

    
    plt.figure()
    plt.imshow(sample_image.detach().numpy().astype(np.uint8))
    plt.show()

    plt.figure()
    plt.imshow(siftimg[:, :, 3], cmap=plt.get_cmap("gist_gray"))
    plt.show()


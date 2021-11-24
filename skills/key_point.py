import math
import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
"""
SIFT
detector = cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
nfeatures: 검출 최대 특징 수
nOctaveLayers: 이미지 피라미드에 사용할 계층 수 default = 3
contrastThreshold: 필터링할 빈약한 특징 문턱 값
edgeThreshold: 필터링할 엣지 문턱 값
sigma: 이미지 피라미드 0 계층에서 사용할 가우시안 필터의 시그마 값
"""
class SIFT(object):
    def __init__(self, mode="sift_img", nfeatures=200, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6, for_test=False):
        self.mode = mode
        self.nfeatures=nfeatures
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold=edgeThreshold
        self.sigma=sigma
        self.for_test=for_test

    def __call__(self, torch_img, kernel_size=(5,5)):
        img = torch_img.detach().numpy().astype(np.uint8)
        if img.shape[0] == 3:
            axis = 0
            gray = np.mean(img, axis=0).astype(np.uint8)
        else:
            axis = 2
            gray = np.mean(img, axis=2).astype(np.uint8)
        sift = cv2.xfeatures2d.SIFT_create(
            nfeatures=self.nfeatures, 
            contrastThreshold=self.contrastThreshold,
            edgeThreshold=self.edgeThreshold,
            sigma=self.sigma
        )

        keypoints, descriptor = sift.detectAndCompute(gray, None)

        if self.for_test:
            # 사람이 시각화해서 볼때 예쁨
            img_draw = cv2.drawKeypoints(img, keypoints, None, \
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return img_draw
        else:
            if "circle" in self.mode:
                sift_img = gray.copy()
                for x in range(1,len(keypoints)):
                    sift_img=cv2.circle(
                        sift_img, 
                        (np.int(keypoints[x].pt[0]),np.int(keypoints[x].pt[1])), 
                        radius=np.int(keypoints[x].size), 
                        color=(0,0,0), thickness=-1
                        )
                sift_img = gray - sift_img
                if self.mode == "blur_circle":
                    blurred = cv2.GaussianBlur(gray.copy(), kernel_size, cv2.BORDER_DEFAULT)
                    sift_img = np.where(sift_img == 0, blurred, sift_img)

            elif self.mode=="default":
                sift_img = np.zeros_like(gray)
                sift_img = cv2.drawKeypoints(sift_img, keypoints,  None, color=(1,1,1,1), \
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)   
                sift_img = np.sum(sift_img, axis=2).astype(np.uint8)
            return torch.cat((torch_img, torch.Tensor(sift_img).unsqueeze(axis)), axis=axis)

"""
해리스 코너 검출
dst = cv2.cornerHarris(src, blockSize, ksize, k, dst, borderType)
src: 입력 이미지, 그레이 스케일
blockSize: 이웃 픽셀 범위
ksize: 소벨 미분 필터 크기
k(optional): 코너 검출 상수 (보토 0.04~0.06)
dst(optional): 코너 검출 결과 (src와 같은 크기의 1 채널 배열, 변화량의 값, 지역 최대 값이 코너점을 의미)
borderType(optional): 외곽 영역 보정 형식
"""
class HarrisCorner(object):
    def __init__(self, for_test=False):
        self.for_test=for_test

    def __call__(self, torch_img):
        img = torch_img.detach().numpy().astype(np.uint8)
        if img.shape[0] == 3:
            axis = 0
            gray = np.mean(img, axis=0).astype(np.uint8)
        else:
            axis = 2
            gray = np.mean(img, axis=2).astype(np.uint8)
        corner = cv2.cornerHarris(gray, 2, 3, 0.04)
        coord = np.where(corner > 0.3* corner.max())
        coord = np.stack((coord[1], coord[0]), axis=-1)
        coner_img = np.zeros(img.shape[:2])
        for x, y in coord:
            cv2.circle(coner_img, (x,y), 5, (1), 1, cv2.LINE_AA)
        
        return torch.concat((torch_img, torch.Tensor(coner_img).unsqueeze(axis)), axis=axis)


if __name__ == '__main__':
    sample_image = Image.open("data/cat_dog/test/dog/dog.12400.jpg")
    sample_image = torch.Tensor(np.array(sample_image))
    # sift = SIFT(for_test=False)
    sift = SIFT(for_test=False,  mode="blur_circle")
    # sift = HarrisCorner(for_test=False)
    siftimg = sift(sample_image)
    print(siftimg.shape)

    
    plt.figure()
    plt.imshow(sample_image.detach().numpy().astype(np.uint8))
    plt.show()

    plt.figure()
    plt.imshow(siftimg[:, :, 3], cmap=plt.get_cmap("gist_gray"))
    plt.show()


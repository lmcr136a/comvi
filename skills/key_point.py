import math
import cv2
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

def sift(img, nfeatures=200, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6, for_test=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create(
        nfeatures=nfeatures, 
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma
    )

    keypoints, descriptor = sift.detectAndCompute(gray, None)

    if for_test:
        img_draw = cv2.drawKeypoints(img, keypoints, None, \
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img_draw
    else:
        sift_img = cv2.drawKeypoints(np.zeros_like(img), keypoints, None, \
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        sift_img = Image.fromarray(sift_img).convert('L')
        return sift_img
    
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

def corner_harris(img, for_test=False):
    # 해리스 코너 검출 ---①
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corner = cv2.cornerHarris(gray, 2, 3, 0.04)
    coord = np.where(corner > 0.3* corner.max())
    coord = np.stack((coord[1], coord[0]), axis=-1)
    coner_img = np.zeros_like(img)
    for x, y in coord:
        cv2.circle(coner_img, (x,y), 5, (0,0,255), 1, cv2.LINE_AA)
    
    return coner_img

if __name__ == '__main__':
    sample_image = Image.open("../data_for_test/testimg.jpg")
    sample_image = np.array(sample_image)
    siftimg = sift(sample_image, for_test=False)
    plt.figure()
    plt.imshow(siftimg)
    plt.show()


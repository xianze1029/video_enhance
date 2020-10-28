# -*- coding: utf-8 -*-
# @Time    : 20-2-6
# @Author  : Song Shuquan, Jiao YingXia
# @File    : enhance_video.py
# @IDE     : Visual Studio Code
"""
Enhance underwater video.
"""
import argparse

import numpy as np
from cv2 import cv2 as cv2


def init_args():
    """

    :param:
    :return:
    """
    parser = argparse.ArgumentParser()#創建ArgumentParser()對象
    parser.add_argument("--video_path", type=str, help="The image file path.")
    parser.add_argument("--denoise",
                        type=str,
                        help="The image denoising mode.",
                        default=None,
                        choices=[None, "median", "gaussian"])
    parser.add_argument("--enhance",
                        type=str,
                        help="The image fuzzy enhancement mode.",
                        default=None,
                        choices=[None, "equalize", "contrast", "fuzzy"])
    parser.add_argument("--sharpen",
                        type=str,
                        help="The image sharpening mode.",
                        default=None,
                        choices=[None, "USM"])
    '''調用add_argument()方法添加函數'''

    return parser.parse_args()#使用parse_args()解析添加的函數



def median_blur(img, ksize=3):#定義函數median_blur(灰色源图像, 滤波模板的尺寸大小为3)
    """
    denoise function using median_blur
    """
    result = cv2.medianBlur(img, ksize)#结果 = cv2.medianBlur(灰色源图像, 核大小)
    return result


def gaussian_blur(img, ksize=(5, 5), sigmaX=0, sigmaY=None):
    '''定義函數gaussian_blur(img, ksize=(5, 5), sigmaX=0, sigmaY=None)表示输入灰色源图像img, ksize是高斯内核大小, sigmaX高斯核函数在X方向上的标准偏差为0'''
    """
    denoise function using gaussian_blur
    """
    result = cv2.GaussianBlur(img, ksize, sigmaX, sigmaY)
    #結果 = cv2.GaussianBlur(灰色源图像, 核大小 ,sigmaX高斯核函数在X方向上的标准偏差, sigmaY高斯核函数在X方向上的标准偏差)
    return result


def equalize_hist(img):
    """
    enhance function using equalize_hist
    """
    # equalize rgb img need to equalize every channel
    (b, g, r) = cv2.split(img)#通道分离，顺序是BGR
    bH = cv2.equalizeHist(b)#彩色图像均衡化, 对每一个通道均衡化
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # merge channels
    result = cv2.merge((bH, gH, rH))#合并单通道成多通道
    return result


def contrast_enhance(img, contrast=60.0):
    """
    enhance function using constrast enhance
    :param: img     :
          : contrast: -100.0 - 100.0
    :return: 
    """
    img = img * 1.0
    thre = img.mean()
    result = img * 1.0
    if contrast <= -255.0:
        result = (result >= 0) + thre - 1
    elif contrast > -255.0 and contrast < 0:
        result = img + (img - thre) * contrast / 255.0
    elif contrast < 255.0 and contrast > 0:
        new_con = 255.0 * 255.0 / (256.0 - contrast) - 255.0
        result = img + (img - thre) * new_con / 255.0
    else:
        mask_1 = img > thre
        result = mask_1 * 255.0
    result = result / 255.0

    mask_1 = result < 0
    mask_2 = result > 1
    result = result * (1 - mask_1)
    result = result * (1 - mask_2) + mask_2
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return result


def fuzzy_enhance(img, Fe=1.0, Fd=120.0):
    #定義fuzzy_enhance(img, 控制參數, 控制參數)函數,Fe與Fd分別是倒数型和指数型模糊因子
    """
    enhance function using fuzzy enhance
    """
    xmax = img.max()
    u = 1 / (1 + (xmax - img) / Fd)**Fe#空间域变换到模糊域

    u1 = 2 * u**2#模糊域增强算子
    u2 = 1 - 2 * (1 - u)**2
    mask = u < 0.5
    n_mask = u >= 0.5
    mask = mask.astype(np.int)
    n_mask = n_mask.astype(np.int)

    u = u1 * mask + u2 * (n_mask)

    img = xmax - (1 / u**(1 / Fe) - 1).dot(Fd)#模糊域变换回空间域
    img[img < 0] = 0
    result = img.astype(np.uint8)
    return result


def USM(img, alpha=1.5, g_ksize=(0, 0), g_sigma=3.0):#定義USM函數
    """
    sharpen function using adobe USM
    """
    img = img * 1.0
    g_out = gaussian_blur(img, g_ksize, g_sigma, g_sigma)

    result = (img - g_out) * alpha + img
    #(原圖 - 高斯模糊圖)类似于使用高通滤波获取细节信息, 产生细节信息掩模 , 将细节信息掩模与原始图像进行叠加，增加边缘的对比度
    result = result / 255.0

    mask_0 = result < 0#原值和低通的差异<0
    mask_1 = result > 1#原值和低通的差异>1

    result = result * (1 - mask_0)
    result = result * (1 - mask_1) + mask_1
    #如果原值和低通的差异的绝对值大于1，则对改点进行所谓的锐化
    result = cv2.normalize(result,
                           dst=None,
                           alpha=0,
                           beta=255,
                           norm_type=cv2.NORM_MINMAX,
                           dtype=cv2.CV_8U)

    return result


def process_img(frame, denoise, enhance, sharpen):
    """
    :param: frame   : current frame
            denoise : the denoise method chose
            enhance : enhance method
            sharpen : shapern method
    :return:
    """
    denoise_function = {"median": median_blur, "gaussian": gaussian_blur}

    enhance_function = {
        "equalize": equalize_hist,
        "contrast": contrast_enhance,
        "fuzzy": fuzzy_enhance
    }

    sharpen_function = {"USM": USM}

    if denoise is not None:
        frame = denoise_function[denoise](frame)

    if enhance is not None:
        frame = enhance_function[enhance](frame)

    if sharpen is not None:
        frame = sharpen_function[sharpen](frame)

    return frame


def process_video(video_path, denoise, enhance, sharpen):
    """

    :param: video_path  : 
    :return:
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('river_result.avi', fourcc, fps, (width, height))
    while cap.isOpened():
        rval, frame = cap.read()
        if rval == True:
            result_frame = process_img(frame,
                                       denoise=denoise,
                                       enhance=enhance,
                                       sharpen=sharpen)
            out.write(result_frame)
            frame_resize = cv2.resize(result_frame,
                                      (int(width / 3), int(height / 3)),
                                      interpolation=cv2.INTER_CUBIC)
            cv2.imshow("example", frame_resize)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    main code
    """
    args = init_args()
    process_video(args.video_path, args.denoise, args.enhance, args.sharpen)

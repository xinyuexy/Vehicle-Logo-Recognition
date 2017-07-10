# -*- coding: utf-8 -*-
from __future__ import division #强制除法为浮点数
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

plate=[] #保存车牌位置

def HSVfilter(img):
    '''HSV过滤蓝色'''
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H,S,V=cv2.split(imgHSV)
    lowerBlue=np.array([100,100,80])
    upperBlue=np.array([130,255,255])
    mask=cv2.inRange(imgHSV,lowerBlue,upperBlue)
    plateImg=cv2.bitwise_and(img,img,mask=mask)
    return mask

def process(img):
    img=cv2.medianBlur(img,5)
    kernel=np.ones((3,3),np.uint8)

    #img=cv2.erode(img,kernel,iterations = 1)
    sobel = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize = 3)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(sobel, element2, iterations = 1)
    erosion = cv2.erode(dilation, element1, iterations = 1)
    dilation2 = cv2.dilate(erosion, element2,iterations = 3)
    #img=cv2.dilate(img,kernel,iterations = 1)
    #img=cv2.Canny(img,100,200)
    return dilation2

def plateDetect(img,img2):
    '''定位车牌区域以及根据车牌粗定位车标'''
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for con in contours:
        x,y,w,h=cv2.boundingRect(con)
        area=w*h
        ratio=w/h
        if ratio>2 and ratio<4 and area>=2000 and area<=25000:
            logo_y1=max(0,int(y-h*3.0))
            logo_y2=y
            logo_x1=x
            logo_x2=x+w
            img_logo=img2.copy()
            logo=img_logo[logo_y1:logo_y2,logo_x1:logo_x2]
            cv2.imwrite('./logo1.jpg',logo)
            cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img2,(logo_x1,logo_y1),(logo_x2,logo_y2),(0,255,0),2)
            global plate
            plate=[x,y,w,h]
            #返回车标粗定位区域
            return logo

def logoDetect(img,imgo):
    '''对粗定位的车标区域进行二次定位'''
    imglogo=imgo.copy()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(2*img.shape[1],2*img.shape[0]),interpolation=cv2.INTER_CUBIC)
    #img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,-3)
    ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #img=cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize = 9)
    img=cv2.Canny(img,100,200)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.dilate(img, element2,iterations = 1)
    img = cv2.erode(img, element1, iterations = 3)
    img = cv2.dilate(img, element2,iterations = 3)

    #查找轮廓
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    tema=0
    result=[]
    for con in contours:
        x,y,w,h=cv2.boundingRect(con)
        area=w*h
        ratio=max(w/h,h/w)
        if area>300 and area<20000 and ratio<2:
            if area>tema:
                tema=area
                result=[x,y,w,h]
                ratio2=ratio
    #计算车标区域在原始图像中的坐标位置,需加上车牌的相对位置
    logo2_X=[int(result[0]/2+plate[0]-3),int(result[0]/2+plate[0]+result[2]/2+3)]
    logo2_Y=[int(result[1]/2+max(0,plate[1]-plate[3]*3.0)-3),int(result[1]/2+max(0,plate[1]-plate[3]*3.0)+result[3]/2)+3]
    cv2.rectangle(img,(result[0],result[1]),(result[0]+result[2],result[1]+result[3]),(255,0,0),2)
    cv2.rectangle(imgo,(logo2_X[0],logo2_Y[0]),(logo2_X[1],logo2_Y[1]),(0,0,255),2)
    print tema,ratio2,result
    logo2=imglogo[logo2_Y[0]:logo2_Y[1],logo2_X[0]:logo2_X[1]]
    cv2.imwrite('./logo2.jpg',logo2)

    return img

if __name__ == '__main__':
    img=cv2.imread('./vehicle/234_2.jpg')
    plateImg=HSVfilter(img)
    plateImg=process(plateImg)
    logo=plateDetect(plateImg,img)
    logo2=logoDetect(logo,img)
    cv2.namedWindow('plate',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('plate',600,400)
    cv2.imshow('plate',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


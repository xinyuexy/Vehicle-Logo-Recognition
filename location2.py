# -*- coding: utf-8 -*-
from __future__ import division #强制除法为浮点数
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

def process(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gau=cv2.GaussianBlur(gray,(5,5),0)
    ret,thre = cv2.threshold(gau, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    med=cv2.medianBlur(thre,5)
    canny=cv2.Canny(thre,100,200)
    #sobel = cv2.Sobel(thre, cv2.CV_8U, 1, 0, ksize = 3)
    dilation=cv2.dilate(canny,element2,iterations = 1)
    dst=cv2.erode(dilation, element1, iterations = 1)
    return dst

def find_row(img,img2):
    T=30
    row=[]
    row_start=0
    row_end=0
    for i in range(img.shape[0]):
        count=0
        for j in range(img.shape[1]-1):
            if img[i,j]!=img[i,j+1]:
                count=count+1
            if count>T:
                row.append(i)
                break
    for i in range(len(row)-2):
        if row[i]==row[i+1]-1 and row[i+1]==row[i+2]-1:
            row_start=row[i]
            break
    for j in range(len(row)-1,,-1):
        if row[j]==row[j+1]-1 and row[j+1]==row[j+2]-1:
            row_end=row[j]
            break
    cv2.line(img,(0,row_start),(img2.shape[1],row_start),(255,0,0),2)
    cv2.line(img,(0,row_end),(img2.shape[1],row_end),(255,0,0),2)
    print row_end

if __name__ == '__main__':
    imgRGB=cv2.imread('D:/pictures/vehicle/2.jpg')
    img=process(imgRGB)
    find_row(img,imgRGB)
    cv2.namedWindow('plate',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('plate',600,400)
    cv2.imshow('plate',imgRGB)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

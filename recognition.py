# -*- coding: utf-8 -*-
from __future__ import division #强制除法为浮点数
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings("ignore")

def preprocess(img):
    '''提取特征前预处理'''
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.medianBlur(img,5)
    img=cv2.equalizeHist(img)
    return img

def BasicLBP(img):
    '''3x3 LBP实现'''
    src=img.copy()
    src=preprocess(src)
    if src.ndim==3:
        src=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    dst=np.zeros((src.shape[0]-2,src.shape[1]-2),dtype=np.uint8)

    for i in range(1,src.shape[0]-1):
        for j in range(1,src.shape[1]-1):
            center=src.item(i,j)
            lbpCode=0
            lbpCode |= (src.item(i-1,j-1) > center) << 7
            lbpCode |= (src.item(i-1,j  ) > center) << 6
            lbpCode |= (src.item(i-1,j+1) > center) << 5
            lbpCode |= (src.item(i  ,j+1) > center) << 4
            lbpCode |= (src.item(i+1,j+1) > center) << 3
            lbpCode |= (src.item(i+1,j  ) > center) << 2
            lbpCode |= (src.item(i+1,j-1) > center) << 1
            lbpCode |= (src.item(i  ,j-1) > center) << 0
            dst.itemset(i-1,j-1,lbpCode)
    return dst

def CircularLBP(img,radius=1,neighbors=8):
    '''圆形LBP实现'''
    src=img.copy()
    src=preprocess(src)
    if src.ndim == 3:
        src=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rows,cols=src.shape
    dst=np.zeros((rows-2*radius,cols-2*radius),dtype=np.uint8)

    for k in range(neighbors):
        #计算采样点对于中心点坐标的偏移量
        rx=radius*math.cos(2.0*math.pi*k/neighbors)
        ry=-radius*math.sin(2.0*math.pi*k/neighbors)
        #为双线性插值做准备,对采样点坐标分别进行上下取整
        x1=math.floor(rx)
        x2=math.ceil(rx)
        y1=math.floor(ry)
        y2=math.ceil(ry)
        #将坐标偏移量映射到0-1之间
        tx=rx-x1
        ty=ry-y1
        #计算双线性插值权重
        w1=(1-tx)*(1-ty)
        w2=tx*(1-ty)
        w3=(1-tx)*ty
        w4=tx*ty
        #循环处理每个像素
        for i in range(radius,rows-radius):
            for j in range(radius,cols-radius):
                #获取中心像素点灰度值
                center=src.item(i,j)
                #根据双线性插值公式计算第k个采样点的灰度值
                neighbor=src.item(i+x1,j+y1)*w1+src.item(i+x1,j+y2)*w2+src.item(i+x2,j+y1)*w3+src.item(i+x2,j+y2)*w4
                dst[i-radius,j-radius]|=(neighbor>center)<<(neighbors-k-1)
    return dst

def getHopTimes(n):
    '''计算某个数的二进制跳变数'''
    count=0
    binaryCode='{0:08b}'.format(n)  #将整数n转换为二进制串
    for i in range(8):
        if binaryCode[i] != binaryCode[(i+1)%8]:
            count=count+1
    return count

def UniformLBP(img,radius=1,neighbors=8):
    '''UniformLBP实现'''
    src=img.copy()
    src=preprocess(src)
    if src.ndim == 3:
        src=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rows,cols=src.shape
    dst=np.zeros((rows-2*radius,cols-2*radius),dtype=np.uint8)

    temp=1
    table=[0]*256
    for i in range(256):
        if getHopTimes(i)<3:
            table[i]=temp
            temp=temp+1
    flag=False #是否进行UniformPattern编码的标志
    for k in range(neighbors):
        if k==neighbors-1:
            flag=True
        rx=radius*math.cos(2.0*math.pi*k/neighbors)
        ry=-radius*math.sin(2.0*math.pi*k/neighbors)
        #为双线性插值做准备,对采样点坐标分别进行上下取整
        x1=math.floor(rx)
        x2=math.ceil(rx)
        y1=math.floor(ry)
        y2=math.ceil(ry)
        #将坐标偏移量映射到0-1之间
        tx=rx-x1
        ty=ry-y1
        #计算双线性插值权重
        w1=(1-tx)*(1-ty)
        w2=tx*(1-ty)
        w3=(1-tx)*ty
        w4=tx*ty
        #循环处理每个像素
        for i in range(radius,rows-radius):
            for j in range(radius,cols-radius):
                #获取中心像素点灰度值
                center=src.item(i,j)
                #根据双线性插值公式计算第k个采样点的灰度值
                neighbor=src.item(i+x1,j+y1)*w1+src.item(i+x1,j+y2)*w2+src.item(i+x2,j+y1)*w3+src.item(i+x2,j+y2)*w4
                dst[i-radius,j-radius]|=(neighbor>center)<<(neighbors-k-1)
                if flag:
                    dst[i-radius,j-radius]=table[dst[i-radius,j-radius]]
    return dst

def BlockLBPH(img,minValue,maxValue,normed=True):
    '''计算一个区域块的LBP特征直方图'''
    #计算直方图bin的数目
    histSize=[maxValue-minValue+1]
    ranges=[minValue,maxValue+1]
    result=cv2.calcHist(img,[0],None,histSize,ranges)
    #归一化
    if normed:
        result=result/(int)(img.shape[0]*img.shape[1])
    return result.reshape(1,-1)

def LBPH(img,numPatterns,grid_x,grid_y,normed=True):
    '''计算LBP特征向量'''
    src=img.copy()
    width=int(src.shape[1]/grid_x)
    height=int(src.shape[0]/grid_y)
    HistLBP=np.zeros((grid_x*grid_y,numPatterns),dtype=np.float32)
    if src.size==0:
        return HistLBP.reshape(1,-1)

    cellIndex=0
    for i in range(grid_x):
        for j in range(grid_y):
            src_cell=src[i*height:(i+1)*height,j*width:(j+1)*width]
            hist_cell=BlockLBPH(src_cell,0,(numPatterns-1),normed)
            HistLBP[cellIndex,:]=hist_cell
            cellIndex=cellIndex+1
    return HistLBP.reshape(1,-1)

if __name__ == '__main__':
    img=cv2.imread('./logo template/ChangAn/20.jpg')
    #lbp=BasicLBP(img)
    Clbp=CircularLBP(img,radius=1,neighbors=6)
    lbph=LBPH(Clbp,64,4,4,normed=True)
    print lbph.shape
    plt.subplot(111)
    plt.plot(lbph.flatten())
    plt.show()
    cv2.namedWindow('LBP',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('LBP',600,600)
    cv2.imshow('LBP',Clbp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

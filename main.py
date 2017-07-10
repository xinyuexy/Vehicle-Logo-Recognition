# -*- coding: utf-8 -*-
from __future__ import division #强制除法为浮点数
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import recognition
import location
import pickle

labelname={'ChangAn':1,'DaZhong':2,'JiangHuai':3,'JinBei':4,'KaiRui':5,'QiRui':6,'XianDai':7}
_labels=[]
_histograms=[]
#训练和预测的参数
_radius=1;_neighbors=6;_grid_x=4;_grid_y=4;_normed=True

def fileRename():
    '''重命名logo template目录下所有车标文件名'''
    rootpath='./logo template/'
    for listdir in os.listdir(rootpath):
        filepath=os.path.join(rootpath,listdir)
        if os.path.isdir(filepath):
            for i,filename in enumerate(os.listdir(filepath)):
                newname='%d'%(i+1)+'.jpg'
                os.rename(filepath+'/'+filename,filepath+'/'+newname)

def buildFeature(trainPath):
    '''建立车标特征模板库,对每一类的图片LBP取均值'''
    global labelname,_labels,_histograms
    rootpath=trainPath

    for listdir in os.listdir(rootpath):
        filepath=os.path.join(rootpath,listdir)
        if os.path.isdir(filepath):
            _labels.append(labelname[listdir])
            list_lbph=[]
            for filename in os.listdir(filepath):
                fileimg=filepath+'/'+filename
                img=cv2.imread(fileimg)
                img=cv2.resize(img,(48,48))
                lbP_img=recognition.CircularLBP(img,radius=_radius,neighbors=_neighbors)
                lbph=recognition.LBPH(lbP_img,int(math.pow(2,_neighbors)),grid_x=_grid_x,grid_y=_grid_y)
                list_lbph.append(lbph)
                #_labels.append(labelname[listdir])
                #_histograms.append(lbph)
            _histograms.append(np.mean(list_lbph,axis=0))
    train_data=[_histograms,_labels]
    #将训练得到的数据持久化到文件中
    pickle.dump(train_data,open('feature.dat','w'))

def predict(img):
    '''对待识别车标进行预测'''
    if len(_histograms)==0:
        print 'model is not build'
        return
    img=cv2.resize(img,(48,48))
    lbP_img=recognition.CircularLBP(img,radius=_radius,neighbors=_neighbors)
    lbph_pre=recognition.LBPH(lbP_img,int(math.pow(2,_neighbors)),grid_x=_grid_x,grid_y=_grid_y)
    minDist=sys.float_info.max
    minClass=-1
    for index in range(len(_histograms)):
        dist=cv2.compareHist(_histograms[index],lbph_pre,cv2.HISTCMP_CHISQR)
        if dist<minDist:
            minDist=dist
            minClass=_labels[index]
            print 'label:%d distance:%f'%(minClass,minDist)
    return minClass,minDist,lbP_img,lbph_pre

if __name__ == '__main__':
    trainPath='./logo template/'
    if os.path.exists('feature.dat'):
        #从feature.dat中恢复数据
        train_data=pickle.load(open('feature.dat','r'))
        _histograms,_labels=train_data
    else:
        print 'feature is building...'
        buildFeature(trainPath)
    print 'feature build finish'
    print labelname

    img=cv2.imread('./logo2.jpg')   #读取待预测车标
    minClass,minDist,lbP_img,lbph_pre=predict(img)
    new_label={v:k for k,v in labelname.items()}  #字典key和value对调
    print 'result:'+new_label[minClass]

    plt.figure()
    plt.subplot(121);plt.imshow(lbP_img,'gray')
    plt.title('LBP')
    plt.subplot(122);plt.plot(lbph_pre.flatten())
    plt.title('LBPH')
    plt.show()




# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:33:42 2015

@author: Yue Lin,SIR
"""
#Normalizar por la media

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import ndimage
from scipy.misc import imresize
from scipy import signal


def getChannel(img):
    size1_size=int(round(img.size[1]/10))
    size2_size=int(round(img.size[0]/10))
    img = imresize(img, ( size1_size,size2_size),interp='bilinear').astype('float')
    img_np = np.array(img)
    x1 = 20; y1 = 20;
    x2 = 25; y2 = 333;
    x3 = 24; y3 = 647;
    w = 336; h = 302;
    im1 = img_np[y1-2:y1+h-2,x1-1:x1+w-1];
    im2 = img_np[y2-2:y2+h-2,x2-1:x2+w-1];
    im3 = img_np[y3-2:y3+h-2,x3-1:x3+w-1];
    I1 = 255*im1.astype('double')/im1.max();
    I2 = 255*im2.astype('double')/im2.max();
    I3 = 255*im3.astype('double')/im3.max();
    RGB=np.empty([h,w,3],dtype=float)
    RGB[:,:,0]=I3
    RGB[:,:,1]=I2
    RGB[:,:,2]=I1
    plt.show()
    plt.imshow(RGB/255)
    return RGB

def cropImage(img,point,extents):
    crop=img[point[1]-extents:point[1]+extents,point[0]-extents:point[0]+extents] 
    return crop

def cropImage2(img,cropSize):
    imgAncho=img.shape[1]
    imgAlto=img.shape[0]    
    beginCorner=[0,0]
    endCorner=[imgAlto,imgAncho]
    cropImg=img[(beginCorner[0]+cropSize[2]):(endCorner[0]+cropSize[3]),(beginCorner[1]+cropSize[0]):(endCorner[1]+cropSize[1])]
    return cropImg    
    
def correlationMatrix(img1,img2):
    matrix=signal.correlate2d(img1,img2,mode="full",boundary="fill",fillvalue=3)
    return matrix    

def refreshCropSizes(cropSizes, cropSize):
    for vector in cropSizes:
        for i in range(len(cropSize)):
            if cropSize[i]>0:
                vector[i+1]=vector[i+1]-cropSize[i]
            elif cropSize[i]<0:
                vector[i-1]=vector[i-1]-cropSize[i]

def generateCropSize(despVector):
    cropSize=[0,0,0,0]
    for i in range(len(despVector)):
        if(despVector[i]>0):
            cropSize[2*i]=despVector[i]
        elif despVector[i]<0:
            cropSize[2*i+1]=despVector[i]
    return cropSize

def main():
    img = Image.open("00029u.png")
    centerPoint=(140,165)    
    extents=15
    image=getChannel(img)
    cropImg=np.empty((extents*2,extents*2,3),dtype=float)   
    cropSizes=[]     
    cropSizes.append([0,0,0,0])
    print image.shape
    for i in range(image.shape[2]):
        cropImg[:,:,i]=cropImage(image[:,:,i],centerPoint,extents)
        cropImg[:,:,i]=cropImg[:,:,i]-np.mean(cropImg[:,:,i])
        plt.show()
        plt.imshow(cropImg[:,:,i])
    channelR=cropImg[:,:,0]
    for i in range(2):
        correlation=correlationMatrix(channelR,cropImg[:,:,i+1])
               
        position=np.where(correlation==np.amax(correlation))
        despVector=[position[0][0]-correlation.shape[0]/2,position[1][0]-correlation.shape[1]/2]
        #print movement        
        cropSize=generateCropSize(despVector)
        refreshCropSizes(cropSizes,cropSize)
        cropSizes.append(cropSize)
    xsize=image[:,:,0].shape[1]-(abs(cropSizes[0][0])+abs(cropSizes[0][1]))
    ysize=image[:,:,0].shape[0]-(abs(cropSizes[0][2])+abs(cropSizes[0][3]))
    alignImage=np.empty((ysize,xsize,3),dtype=float)
    for i in range(3):
        alignImage[:,:,i]=cropImage2(image[:,:,i],cropSizes[i])
    plt.show()
    plt.imshow(alignImage/255)
    
        #print np.amax(correlation)
        #plt.show()
        #plt.imshow(correlation)        
        
        #plt.imshow(ndimage.convolve(cropImg[:,:,i],correlation,mode="constant",cval=0.0))        
        #plt.show()
        #plt.imshow(cropImg/255)
main()

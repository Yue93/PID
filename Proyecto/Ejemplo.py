# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 16:48:35 2015

@author: SIR
"""
from scipy import ndimage
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color, data
import sys
import os
from PIL import Image
from pylab import ion, ioff
from scipy import fftpack
from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt
#http://www-rohan.sdsu.edu/doc/matlab/toolbox/images/deblurr9.html
#http://zoi.utia.cas.cz/fastdeconv.html
    
def gaussiana(sigma):
    oscRange=5*sigma/2
    filtro=np.empty((2*oscRange+1,2*oscRange+1),dtype=float)
    x0=oscRange+1
    y0=x0
    minxy=x0-oscRange
    maxxy=x0+oscRange
    for i in range(minxy,maxxy):
        componente1=((math.pow(i-x0,2))/(2*math.pow(sigma,2)))  
        for j in range(minxy,maxxy):
            componente2=((math.pow(j-y0,2))/(2*math.pow(sigma,2)))
            exponente=componente1+componente2
            filtro[i-minxy,j-minxy]=math.exp(-exponente)
    filtro=filtro/np.sum(filtro)
    return filtro

def getEdge(img,mode=0):
    return {0: sobel(img),
        1:scharr(img),
        2:prewitt(img),
        3:roberts(img)}.get(mode, sobel(img))

def blurImage(image,filtro):
    return ndimage.convolve(image,filtro,mode="constant",cval=0.0)
    

def deblurImage(blurImg,iteration,mode=0):
    edge=getEdge(blurImg,mode)
    stDesv=np.std(edge)
    gradx, grady=np.gradient(blurImg)
    deblurImg=np.copy(blurImg)
    print "Gradiente antes", np.sum(gradx+grady)    
    for i in range(iteration):
        for j in range(deblurImg.shape[0]):
            for k in range(deblurImg.shape[1]):
                gain=abs(stDesv*gradx[j,k])+abs(stDesv*grady[j,k])
                deblurImg[j,k]=deblurImg[j,k]+gain
        edge=getEdge(deblurImg,mode)
        stDesv=np.std(edge)
    gradx2, grady2=np.gradient(deblurImg)
    print "Gradiente despues", np.sum(gradx2+grady2)
    return deblurImg

def main():
    fftsize=1024
    sigma=4
    iteration=1
    #img=io.imread("torre.jpg")
    #print "Shape: ",img.shape
    #im = np.mean(img,axis=2)/255.    
    #im_fft=fftpack.fft2(im,(fftsize, fftsize))
    #F = np.log(1+np.abs(im_fft))
    #recovered = np.real(fftpack.ifft2(im_fft))
    #plt.show()
    #plt.imshow(im, cmap='gray')
    #plt.title('Imagen en gris')
    #blurImg(im,fftsize)

    img = io.imread("torre.jpg")
    
    grayImg=color.rgb2gray(img)
    
    filtro=gaussiana(sigma)
    
    blurImg=blurImage(grayImg,filtro)
    
    deblurImg=deblurImage(blurImg,iteration,0)
    
    plt.figure(0)
    plt.subplot(2,2,1)
    plt.imshow(grayImg, cmap="gray")
    plt.title("Original Image")
    
    plt.subplot(2,2,2)
    plt.imshow(blurImg, cmap="gray")
    plt.title("Blur Image")

    plt.subplot(2,2,3)
    plt.imshow(deblurImg, cmap="gray")
    plt.title("Image after "+str(iteration)+" iter")    
    
main()

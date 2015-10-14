# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:06:32 2015

@author: enrique
"""
from scipy import misc
from scipy import fftpack
from scipy import ndimage
from IPython.html.widgets import interact
from skimage import io
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os

def gaussiana(oscRange,sigma):
    #oscRange=6*sigma/2
    filtro=np.empty((2*oscRange+1,2*oscRange+1),dtype=float)
    x0=oscRange+1
    y0=x0
    minxy=x0-oscRange
    maxxy=x0+oscRange
    print "X0,y0", x0,y0
    print "Filter size:",filtro.shape
    for i in range(minxy,maxxy):
        componente1=((math.pow(i-x0,2))/(2*math.pow(sigma,2)))  
        for j in range(minxy,maxxy):
            componente2=((math.pow(j-y0,2))/(2*math.pow(sigma,2)))
            exponente=componente1+componente2
            filtro[i-minxy,j-minxy]=math.exp(-exponente)
    filtro=filtro/np.sum(filtro)
    return filtro
    
def lowFilter(img,filtro):
    print "Tamany",img.shape
    imgConv=np.empty((img.shape[0],img.shape[1],img.shape[2]),dtype=float)
    for i in range(img.shape[2]):
        imgConv[:,:,i]=ndimage.convolve(img[:,:,i],filtro,mode="constant",cval=0.0)
        #if(np.amin(imgConv[:,:,i])<0.0):
         #   imgConv[:,:,i]=imgConv[:,:,i]+abs(np.amin(imgConv[:,:,i])*2)
          #  imgConv[:,:,i]=imgConv[:,:,i]/np.sum(imgConv[:,:,i])
    return imgConv
	
def highFilter(img,filtro):
	
     highConvImg = 1-lowFilter(img,filtro)
     return highConvImg

     #highConvImg=np.empty((img.shape[0],img.shape[1],img.shape[2]),dtype=float)
	#for i in range(img.shape[2]):
     #    highConvImg[:,:,i]=img[:,:,i]-lowConvImg[:,:,i]
     #    if(np.amin(highConvImg[:,:,i])<0.0):
     #        highConvImg[:,:,i]=((highConvImg[:,:,i]-(np.amin(highConvImg[:,:,i])))/((np.amax(highConvImg[:,:,i]))-(np.amin(highConvImg[:,:,i]))))
             #highConvImg[:,:,i]=highConvImg[:,:,i]/np.sum(highConvImg[:,:,i])
	

def TDFourier():
    
    raiz=os.getcwd()
    
    im = io.imread(raiz+"\human.png")
    for i in range(im.shape[2]):
        fftsize=512            
        im_fft = fftpack.fft2(im[:,:,i], (fftsize, fftsize))
        hs = 50
        fil = fspecial('gaussian', hs*2+1, 10)
        fil_fft = fft2(fil, fftsize, fftsize)    
        im_fil_fft = im_fft * fil_fft[:,:,i]
        im_fil[:,:,i] = ifft2(im_fil_fft)
        im_fil[:,:,i] = im_fil(1 + hs:size(im[:,:,1],1)+hs,1 + hs:size(im[:,:,1], 2)+hs)
    
    
    plt.show()
    plt.imshow(im_fil)
    plt.colorbar()
    
    
    #im_fil = im_fil(1 + hs:size(im,1)+hs,1 + hs:size(im, 2)+hs)
    #im_fil = im_fil(1+hs:im.shape[0]+hs,1+hs:im.shape[1]+hs)
    
    
    
TDFourier()
    
    
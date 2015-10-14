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

def gaussiana(tamanyo):
    [xx,yy]=np.meshgrid(np.linspace(-4,4,SZ),np.linspace(-4,4,SZ))
    gaussian = np.exp(-0.5*(xx*xx+yy*yy))
    gaussian = gaussian/np.sum(gaussian)
        
    return gaussian
    
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
    
    im = mpimg.imread(raiz+"\human.png")
    fftsize=512
    
    SZ = 50
    [xx,yy]=np.meshgrid(np.linspace(-4,4,SZ),np.linspace(-4,4,SZ))
    gaussian = np.exp(-0.5*(xx*xx+yy*yy))
    gaussian = gaussian/np.sum(gaussian)
    hs=np.floor(SZ/2.)
    fil = gaussian
    plt.show()
    plt.imshow(gaussian)
    plt.colorbar()
    
    im_crop=np.empty((im.shape[0],im.shape[1],im.shape[2]),dtype=float)
        
    for i in range(im.shape[2]):
                    
        im_fft = fftpack.fft2(im[:,:,i], (fftsize, fftsize))
        
        fil_fft = fftpack.fft2(fil, (fftsize, fftsize))    

        im_fil_fft = im_fft * fil_fft
        #im_fil[:,:,i] = fftpack.ifft2(im_fil_fft)
        im_fil = np.real(fftpack.ifft2(im_fil_fft))
        
        im_crop[:,:,i] = im_fil[hs:im[:,:,i].shape[0]+hs, hs:im[:,:,i].shape[1]+hs]        
        #im_fil[:,:,i] = im_fil[1+hs:size(im,1)+hs,1+hs:size(im,2)+hs]
    plt.show()
    plt.imshow(im_crop)
    plt.colorbar()
    
    
    
    
    #im_fil = im_fil(1 + hs:size(im,1)+hs,1 + hs:size(im, 2)+hs)
    #im_fil = im_fil(1+hs:im.shape[0]+hs,1+hs:im.shape[1]+hs)
    
    
    
TDFourier()
    
    
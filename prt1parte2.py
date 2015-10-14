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
from skimage import color
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
	

def gaussian(sz):
    [xx,yy]=np.meshgrid(np.linspace(-4,4,sz),np.linspace(-4,4,sz))
    gaussian = np.exp(-0.5*(xx*xx+yy*yy))
    gaussian = gaussian/np.sum(gaussian)
    return gaussian

def TDFourier(img,fftsize,hs,fil):
    im_crop=np.empty((img.shape[0],img.shape[1]),dtype=float)
                    
    im_fft = fftpack.fft2(img, (fftsize, fftsize))
    
    fil_fft = fftpack.fft2(fil, (fftsize, fftsize))    

    im_fil_fft = im_fft * fil_fft
    #im_fil[:,:,i] = fftpack.ifft2(im_fil_fft)
    im_fil = np.real(fftpack.ifft2(im_fil_fft))
    
    im_crop= im_fil[hs:img.shape[0]+hs, hs:img.shape[1]+hs]        
        #im_fil[:,:,i] = im_fil[1+hs:size(im,1)+hs,1+hs:size(im,2)+hs]
    plt.show()
    plt.imshow(im_crop)
    plt.colorbar()
    return im_crop


def main():
    raiz=os.getcwd()
    
    #imColor = mpimg.imread(raiz+"\human.png")
    imColor = mpimg.imread(raiz+"\Accelrys.png")
    imGray = color.rgb2gray(mpimg.imread(raiz+"\cbs.png"))
    fftsize=1024
    
    SZ = 50
    hs=np.floor(SZ/2.)
    
    gaussiana=gaussian(SZ)
    print np.sum(gaussiana)
    highFilter=(1-gaussiana)/np.sum(1-gaussiana)
    print np.sum(highFilter)
    lowColorImage=np.empty((imColor.shape[0],imColor.shape[1],imColor.shape[2]),dtype=float)
    convImg=np.empty((imColor.shape[0],imColor.shape[1],imColor.shape[2]),dtype=float)
    for i in range(imColor.shape[2]):
        lowColorImage[:,:,i]=TDFourier(imColor[:,:,i],fftsize,hs,gaussiana)
    highGrayImage=TDFourier(imGray,fftsize,hs,highFilter)
    for i in range(imColor.shape[2]):
        suma=lowColorImage[:,:,i]+highGrayImage
        convImg[:,:,i]=(suma-np.amin(suma))/(np.amax(suma)-np.amin(suma))
    plt.show()
    plt.imshow(lowColorImage)
    plt.colorbar()
    plt.show()
    plt.imshow(highGrayImage).set_cmap('gray')
    plt.colorbar()
    plt.show()
    plt.imshow(convImg)
    plt.colorbar()
main()
    #im_fil = im_fil(1 + hs:size(im,1)+hs,1 + hs:size(im, 2)+hs)
    #im_fil = im_fil(1+hs:im.shape[0]+hs,1+hs:im.shape[1]+hs)

    
    
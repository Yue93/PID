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
#http://www-rohan.sdsu.edu/doc/matlab/toolbox/images/deblurr9.html
#http://zoi.utia.cas.cz/fastdeconv.html

def blurImg(img,fftsize):
    im_fft=fftpack.fft2(img,(fftsize, fftsize))
    SZ = 50
    [xx,yy]=np.meshgrid(np.linspace(-4,4,SZ),np.linspace(-4,4,SZ))
    gaussian = np.exp(-0.5*(xx*xx+yy*yy))
    fil = gaussian/np.sum(gaussian)
    fil_fft = fftpack.fft2(fil, (fftsize, fftsize)) 
    im_fil_fft=im_fft*fil_fft
    im_fil = np.real(fftpack.ifft2(im_fil_fft))
    hs=np.floor(SZ/2.)
    im_crop = im_fil[hs:img.shape[0]+hs, hs:img.shape[1]+hs]
    F=fftpack.fft2(im_crop,(fftsize, fftsize))
    plt.show()
    plt.imshow(im_crop,cmap='gray')
    I=F/fil_fft
    I=np.where(np.abs(fil_fft)<1e-3,0,I)
    img_reconstructed=np.real(fftpack.ifft2(I))
    plt.show()
    plt.imshow(img_reconstructed[:img.shape[0],:img.shape[1]])

def main():
    fftsize=1024
    #img=io.imread("torre.jpg")
    #print "Shape: ",img.shape
    #im = np.mean(img,axis=2)/255.    
    #im_fft=fftpack.fft2(im,(fftsize, fftsize))
    #F = np.log(1+np.abs(im_fft))
    #recovered = np.real(fftpack.ifft2(im_fft))
    a=np.zeros((3,3),dtype=float)
    b=np.ones((3,3),dtype=float)
    a[0][0]=2    
    a[0][1]=3
    a[0][2]=1
    #plt.show()
    #plt.imshow(im, cmap='gray')
    #plt.title('Imagen en gris')
    #blurImg(im,fftsize)

    imA = io.imread("torre.jpg")


    im = np.mean(imA,2)/255.
    
    fftsize = 1024
    im_fft = fftpack.fft2(im, (fftsize, fftsize))
    
    #Complementary of a Gaussian filter
    SZ = 1024
    sigma = 0.25
    [xx,yy]=np.meshgrid(np.linspace(-4,4,SZ),np.linspace(-4,4,SZ))
    gaussian = np.exp(-0.5*(xx*xx+yy*yy)/(sigma*sigma))
    fil =1.-fftpack.fftshift(gaussian/np.max(gaussian))
    
    fil_fft =  fil
    
    im_fil_fft = im_fft * fil_fft
    
    im_fil = np.real(fftpack.ifft2(im_fil_fft))
    
    hs=np.floor(SZ/2.)
    #Careful with the crop. Because we work directly in the Fourier domain there is no padding.
    im_crop = im_fil[0:im.shape[0], 0:im.shape[1]]     
    F=fftpack.fft2(im_crop,(1024,1024))
    H=fil_fft
    tol= 1e-2
    I = F/H
    print np.min(I)
    I=np.where(np.abs(H)<tol,0,I)
    i_reconstructed = np.real(fftpack.ifft2(I))
    plt.imshow(i_reconstructed[:im.shape[0],:im.shape[1]],cmap="gray")
    #imgColor=color.gray2rgb(i_reconstructed[:im.shape[0],:im.shape[1]])
    #plt.show()
    #plt.imshow(imgColor)
    
main()

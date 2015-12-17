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

def inverseFilter(img,fftsize):
    im = np.mean(img,2)/255.
    
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

def normalize(img):
    img[:,:]=((img[:,:]-(np.amin(img[:,:])))/((np.amax(img[:,:]))-(np.amin(img[:,:]))))

def getEdge(img,mode=0):
    return {0: sobel(img),
        1:scharr(img),
        2:prewitt(img),
        3:roberts(img)}.get(mode, sobel(img))

def blurImage(image,filtro):
    return ndimage.convolve(image,filtro,mode="constant",cval=0.0)
    

def deblurImage(blurImg,iteration,mode=0):
    niter=0
    edge=getEdge(blurImg*255,mode)
    stDesv=np.std(edge)
    grady, gradx=np.gradient((blurImg*255))
    deblurImg=np.copy(blurImg)
    normalizar=False
    #print "Gradiente antes", np.sum(gradx+grady) 
    desv=np.std(deblurImg)
    extraGain=1.0
    while(niter<iteration):
        desv=np.std(deblurImg)
        #Dprint "Desviacion estandar borrosa", desv
        for j in range(deblurImg.shape[0]-1):
            for k in range(deblurImg.shape[1]-1):
                gain=gradx[j,k]*stDesv+grady[j,k]*stDesv
                if(gain<extraGain):
                    extraGain=gain
                deblurImg[j,k]=deblurImg[j,k]+gain
                if(deblurImg[j,k]<0.0 or deblurImg[j,k]>255.0):
                    normalizar=True
        deblurImg=extraGain/10.0+deblurImg
        if normalizar:
            normalize(deblurImg)
        
        edge=getEdge(deblurImg,mode)
        stDesv=np.std(edge)
        niter=niter+1
    gradx2, grady2=np.gradient(deblurImg)
    return deblurImg

def main():
    fftsize=1024
    sigma=4
    iteration=25

    img = io.imread("torre.jpg")
    
    grayImg=color.rgb2gray(img)
    
    filtro=gaussiana(sigma)
    
    blurImg=blurImage(grayImg,filtro)

    plt.show()
    plt.imshow(blurImg, cmap="gray")
    plt.title("Blur Image")    
    
    inverseFilter(img,fftsize)
    
    deblurImg=deblurImage(blurImg,iteration,0)
    deblurImg=deblurImg+blurImg
    normalize(deblurImg)
    print "Max",np.amax(deblurImg)
    hist, bin_edges=np.histogram(deblurImg)
    print "Histograma", hist
    print "Bin edges",bin_edges

    plt.show()    
    plt.imshow(deblurImg, cmap="gray")
    plt.title("Image after "+str(iteration)+" iter")    
    
main()

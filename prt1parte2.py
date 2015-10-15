# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:06:32 2015

@author: Yue Lin, enrique
"""
from scipy import misc
from scipy import fftpack
from scipy import ndimage
from IPython.html.widgets import interact
from skimage import io
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


 #Funcion para crear el filtro gaussiana en el dominio de las frecuencias
def gaussian(sz):
    [xx,yy]=np.meshgrid(np.linspace(-4,4,sz),np.linspace(-4,4,sz))
    gaussian = np.exp(-0.5*(xx*xx+yy*yy))
    gaussian = gaussian/np.sum(gaussian)
    return gaussian

#Funciona para aplicar la transformada de fourier a la imagen
def TDFourier(img,fftsize,hs,fil):
    im_crop=np.empty((img.shape[0],img.shape[1]),dtype=float)
                    
    im_fft = fftpack.fft2(img, (fftsize, fftsize))
    
    fil_fft = fftpack.fft2(fil, (fftsize, fftsize))    

    im_fil_fft = im_fft * fil_fft

    im_fil = np.real(fftpack.ifft2(im_fil_fft))
    
    im_crop= im_fil[hs:img.shape[0]+hs, hs:img.shape[1]+hs]        

    #plt.show()
    #plt.imshow(im_crop)
    #plt.colorbar()
    return im_crop

#Funcion principal
def main():
    raiz=os.getcwd()
    imColor = mpimg.imread(raiz+"\Accelrys.png")
    #imGray = color.rgb2gray(mpimg.imread(raiz+"\Cbs.png"))
    imColor2 = mpimg.imread(raiz+"\Cbs.png")
    fftsize=1024
    
    plt.show()
    plt.imshow(imColor)
    plt.colorbar()  
    plt.title("First image")
    
    plt.show()
    plt.imshow(imColor2)
    plt.colorbar()  
    plt.title("Second image")
    
    SZ = 50
    hs=np.floor(SZ/2.)
    
    gaussiana=gaussian(SZ)
    highFilter=(1-gaussiana)/np.sum(1-gaussiana)
    
    lowColorImage=np.empty((imColor.shape[0],imColor.shape[1],imColor.shape[2]),dtype=float)
    convImg=np.empty((imColor.shape[0],imColor.shape[1],imColor.shape[2]),dtype=float)
    #highGrayImage=TDFourier(imGray,fftsize,hs,highFilter)
    highColorImage=np.empty((imColor2.shape[0],imColor2.shape[1],imColor2.shape[2]),dtype=float)
    
    for i in range(imColor.shape[2]):
        highColorImage[:,:,i]=TDFourier(imColor2[:,:,i],fftsize,hs,highFilter)
        lowColorImage[:,:,i]=TDFourier(imColor[:,:,i],fftsize,hs,gaussiana)
        suma=lowColorImage[:,:,i]+highColorImage[:,:,i]
        convImg[:,:,i]=(suma-np.amin(suma))/(np.amax(suma)-np.amin(suma))
        
        
    plt.show()
    plt.imshow(lowColorImage)
    plt.colorbar()
    plt.title("LP rgb image")
    
    plt.show()
    plt.imshow(highColorImage)
    plt.colorbar()
    plt.title("HP rgb image")    
    
    plt.show()
    plt.imshow(convImg)
    plt.colorbar()
    plt.title("Hybrid image")
main()

    
    
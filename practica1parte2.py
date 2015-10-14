# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:05:35 2015

@author: enrique
"""
from scipy import fftpack
from scipy import ndimage
from IPython.html.widgets import interact
from skimage import io
#from practica1 import gaussiana
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
from PIL import Image

def gaussiana(sigma):
    oscRange=6*sigma/2
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

def parte2():
    raiz=os.getcwd()
    im = io.imread(raiz+"\human.png")
    fftsize = 1024
    im_fft = fftpack.fft2(im, (fftsize, fftsize))
    
    SZ = 50
    [xx,yy]=np.meshgrid(np.linspace(-4,4,SZ),np.linspace(-4,4,SZ))
    #gaussian = np.exp(-0.5*(xx*xx+yy*yy))
    #gaussian = gaussian/np.sum(gaussian)
    gaussian=gaussiana(9)
    fil_fft = fftpack.fft2(fil, (fftsize, fftsize))
    im_fil_fft = im_fft * fil_fft
    im_fil = fftpack.ifft2(im_fil_fft)
    hs=np.floor(SZ/2.)
    #im_fil = im_fil(hs:im.shape[0]+hs, hs:im.shape[1]+hs)
    im_fil = im_fil(0:im.shape[0]+hs,0:im.shape[1]+hs)
    
    plt.show()
    plt.imshow(im_fil)
    plt.colorbar()
        
parte2()

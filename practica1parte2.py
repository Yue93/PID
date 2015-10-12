# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:05:35 2015

@author: enrique
"""
from scipy import *
from IPython.html.widgets import interact
from skimage import io
from practica1 import *
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
from PIL import Image


def parte2():
    raiz=os.getcwd()
    image = io.imread(raiz+"\human.png")
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
    im_fil = im_fil(hs:im.shape[0]+hs, hs:im.shape[1]+hs)
    
    
    
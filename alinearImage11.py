# -*- coding: utf-8 -*-
"""
Created on Thu Oct 01 16:33:55 2015

@author: SIR
"""

from scipy import ndimage
import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from PIL import Image
#import Image

#Sugerencia:
#utilizar el siguiente codigo para reducir y alinear previamente las imagenes
def alineacion():
    #raiz="C:\Users\SIR\Desktop\UNIVERSIDAD DE BARCELONA\Curso 15-16\Procesamiento de imagenes\Practica\Practica1"
    #LOAD IMAGES
    raiz=os.getcwd()    
    img1 = Image.open(raiz+"\human.png")
    img2 = Image.open(raiz+"\cat.png")
    #RESIZE
    size1=int(round(img1.size[0]*(4/3.)));
    size2=int(round(img1.size[1]*(4/3.)));
    img1=img1.resize((size1,size2), Image.ANTIALIAS)
    #CROP
    left = 50
    top = 50
    right = 51+img2.size[0]-1
    bottom = 51+img2.size[1]-1
    img1=img1.crop((left, top, right, bottom))
    # VISUALIZATION ---------------
    plt.figure(1)
    plt.subplot(131)
    imgplot1=plt.imshow(img1)
    imgplot1.set_cmap('gray')
    plt.subplot(132)
    imgplot1=plt.imshow(img2)
    imgplot1.set_cmap('gray')
    
    return img1,img2
    
alineacion()
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 01 15:31:20 2015

@author: SIR
"""

#Documentos a entregar:Un pdf de breve explicacion y el código
#Sigma =9 y Masksize= 6 * sigma + 1
#Centro de la mascara:Masksize= 6 * sigma / 2
#La mascara tiene que estar normalizada.Suma de todos los pixeles=0
#Imagen filtrado pasabajo=Aplicar mediante la convolucion un filtro gausiana a la imagen original
#Imagen filtrado pasaalto=Imagen original-imagen filtrado pasabajo
#convol de la libreria scipy o imfilter para hacer la convolucion
#Amplitud= 1/nº de pixel


#ndimage.convolve(I1, If, mode, 'constant', cval=0.0)

from scipy import ndimage
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import sys
import os

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
    
def lowFilter(img,filtro):
    print "Tamany",img.shape
    imgConv=np.empty((img.shape[0],img.shape[1],img.shape[2]),dtype=float)
    for i in range(img.shape[2]):
        imgConv[:,:,i]=ndimage.convolve(img[:,:,i],filtro,mode="constant",cval=0.0)
        #if(np.amin(imgConv[:,:,i])<0.0):
         #   imgConv[:,:,i]=imgConv[:,:,i]+abs(np.amin(imgConv[:,:,i])*2)
          #  imgConv[:,:,i]=imgConv[:,:,i]/np.sum(imgConv[:,:,i])
    return imgConv
	
def highFilter(img,lowConvImg):
	highConvImg=np.empty((img.shape[0],img.shape[1],img.shape[2]),dtype=float)
	for i in range(img.shape[2]):
         highConvImg[:,:,i]=img[:,:,i]-lowConvImg[:,:,i]
         if(np.amin(highConvImg[:,:,i])<0.0):
             highConvImg[:,:,i]=((highConvImg[:,:,i]-(np.amin(highConvImg[:,:,i])))/((np.amax(highConvImg[:,:,i]))-(np.amin(highConvImg[:,:,i]))))
             #highConvImg[:,:,i]=highConvImg[:,:,i]/np.sum(highConvImg[:,:,i])
	return highConvImg

def imgHibrida():
    raiz=os.getcwd()
    filtro=gaussiana(6)
    gato=mpimg.imread(raiz+"\cat.png")
    humano=mpimg.imread(raiz+"\human.png")

    gatoConv=lowFilter(gato,filtro)
    humanoConv=lowFilter(humano,filtro)
    humanoHighConv=highFilter(humano,humanoConv)
    #humanoHighConv=humano-humanoConv    
    #print humanoConv
    print humano.shape, " ",humanoHighConv.shape
    print gato.shape, "  ",gatoConv.shape
    plt.show()
    plt.imshow(humanoHighConv)
    plt.colorbar()
    
    plt.show()
    plt.imshow(gatoConv)
    #print humanoHighConv[:,:,0]
    #print filtro
    #plt.show()
    #plt.imshow(filtro)
imgHibrida()
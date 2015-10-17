# -*- coding: utf-8 -*-
"""
Created on Thu Oct 01 15:31:20 2015

@author: Yue Lin, Enrique
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
from PIL import Image

#Funcion para alinear la imagen del humano con la imagen del gato
def alinear(img1,img2):
    size1=int(round(img1.size[0]*(4/3.)));
    size2=int(round(img1.size[1]*(4/3.)));
    img1=img1.resize((size1,size2), Image.ANTIALIAS)
    #CROP
    left = 50
    top = 50
    right = 51+img2.size[0]-1
    bottom = 51+img2.size[1]-1
    img1=img1.crop((left, top, right, bottom))
    img1.save("humanAlign.png")

#Funciona para crear el filtro gaussiana de baja frecuencia
def gaussiana(sigma):
    oscRange=6*sigma/2
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
    
#Funcion para realizar la convolucion con un filtro de baja frecuencia
def lowFilter(img,filtro):
    imgConv=np.empty((img.shape[0],img.shape[1],img.shape[2]),dtype=float)
    for i in range(img.shape[2]):
        imgConv[:,:,i]=ndimage.convolve(img[:,:,i],filtro,mode="constant",cval=0.0)
    return imgConv
	
#Funcion para realizar la convolucion de una imagen con un filtro de alta frecuencia
def highFilter(img,lowConvImg):
	highConvImg=np.empty((img.shape[0],img.shape[1],img.shape[2]),dtype=float)
	for i in range(img.shape[2]):
         highConvImg[:,:,i]=img[:,:,i]-lowConvImg[:,:,i]
         if(np.amin(highConvImg[:,:,i])<0.0):
             highConvImg[:,:,i]=((highConvImg[:,:,i]-(np.amin(highConvImg[:,:,i])))/((np.amax(highConvImg[:,:,i]))-(np.amin(highConvImg[:,:,i]))))
	return highConvImg

#Funcon para normalizar una imagen
def normalizar(img):
    for i in range(img.shape[2]):
        img[:,:,i]=((img[:,:,i]-(np.amin(img[:,:,i])))/((np.amax(img[:,:,i]))-(np.amin(img[:,:,i]))))

#Funcion principal donde llama a las otras funciones para crear la imagen hibrida
def imgHibrida():
    raiz=os.getcwd()
    filtro=gaussiana(9)
    alinear(Image.open(raiz+"\human.png"),Image.open(raiz+"\cat.png"))
    gato=mpimg.imread(raiz+"\cat.png")
    print gato.shape
    humano=mpimg.imread(raiz+"\humanAlign.png")

    gatoConv=lowFilter(gato,filtro)
    humanoConv=lowFilter(humano,filtro)
    gatoHighConv=highFilter(gato,gatoConv)

    plt.show()
    plt.imshow(gatoHighConv)
    plt.colorbar()
    plt.title("Gat(Convolution with hp)")
    
    plt.show()
    plt.imshow(humanoConv)
    plt.colorbar()
    plt.title("Human(Convolution with lp)")
    
    finalImage=gatoHighConv+humanoConv
    normalizar(finalImage)
    plt.show()
    plt.imshow(finalImage)
    plt.colorbar()
    plt.title("Hybrid Image")
    mpimg.imsave("HybridImage1.png",finalImage)
imgHibrida()
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
def imgHibrida():
    raiz="C:\Users\SIR\Desktop\UNIVERSIDAD DE BARCELONA\Curso 15-16\Procesamiento de imagenes\Practica\Practica1"
    filtro=gaussiana(9)
    gato=mpimg.imread(raiz+"\cat.png")
    humano=mpimg.imread(raiz+"\human.png")
    
    humanoR=humano[:,:,0]
    humanoG=humano[:,:,1]
    humanoB=humano[:,:,2]
    
    humanoRConvBajo=ndimage.convolve(humanoR,filtro,mode="constant",cval=0.0)
    humanoGConvBajo=ndimage.convolve(humanoG,filtro,mode="constant",cval=0.0)
    humanoBConvBajo=ndimage.convolve(humanoB,filtro,mode="constant",cval=0.0)
    
    humanoConv=np.empty((323,285,3),dtype=float)
    humanoConv[:,:,0]=humanoRConvBajo
    humanoConv[:,:,1]=humanoGConvBajo
    humanoConv[:,:,2]=humanoBConvBajo
    #print humanoConv
    plt.show()
    plt.imshow(humanoConv)
    
    
    
    gatoR=gato[:,:,0]
    gatoG=gato[:,:,1]
    gatoB=gato[:,:,2]   
    
    gatoRConvBajo=ndimage.convolve(gatoR,filtro,mode="constant",cval=0.0)
    gatoRConvBajo=gatoRConvBajo+abs(np.amin(gatoRConvBajo))*2
    gatoRConvBajo=gatoRConvBajo/np.sum(gatoRConvBajo)
    
    gatoGConvBajo=ndimage.convolve(gatoG,filtro,mode="constant",cval=0.0)
    gatoGConvBajo=gatoGConvBajo+abs(np.amin(gatoRConvBajo))*2
    gatoGConvBajo=gatoGConvBajo/np.sum(gatoGConvBajo)
    
    gatoBConvBajo=ndimage.convolve(gatoB,filtro,mode="constant",cval=0.0)
    gatoBConvBajo=gatoBConvBajo+abs(np.amin(gatoRConvBajo))*2
    gatoBConvBajo=gatoBConvBajo/np.sum(gatoBConvBajo)
    
    gatoRConvAlto=gatoR-gatoRConvBajo
    gatoGConvAlto=gatoG-gatoGConvBajo
    gatoBConvAlto=gatoB-gatoBConvBajo
    
    gatoConv=np.empty((352,288,3),dtype=float)
    gatoConv[:,:,0]=gatoRConvAlto
    gatoConv[:,:,1]=gatoGConvAlto
    gatoConv[:,:,2]=gatoBConvAlto
    plt.show()
    plt.imshow(gatoConv)
    #print filtro
    #plt.show()
    #plt.imshow(filtro)
    
imgHibrida()
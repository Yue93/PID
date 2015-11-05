# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 19:02:43 2015

@author: Yue Lin y Enrique
"""

#Gradiente y contornos.
#Señalar el punto comun de los barquitos manualmente

import math
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy import ndimage
from scipy.misc import imresize
from scipy import signal
from skimage.exposure import equalize_hist
import time

#Funcion para generar la campana gaussiana
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

#Aplicamos la convolucion a una imagen monocanal que le pasamos    
def lowFilter(img,filtro):
    imgConv=np.empty((img.shape[0],img.shape[1]),dtype=float)
    imgConv=ndimage.convolve(img,filtro,mode="constant",cval=0.0)
    return imgConv

#Funcion para general las canales de una imagen
#que leemos con la libreria Image
def getChannel(img):
    size1_size=int(round(img.size[1]/10))
    size2_size=int(round(img.size[0]/10))
    img = imresize(img, ( size1_size,size2_size),interp='bilinear').astype('float')
    img_np = np.array(img)
    x1 = 20; y1 = 20;
    x2 = 25; y2 = 333;
    x3 = 24; y3 = 647;
    w = 336; h = 302;
    im1 = img_np[y1-2:y1+h-2,x1-1:x1+w-1];
    im2 = img_np[y2-2:y2+h-2,x2-1:x2+w-1];
    im3 = img_np[y3-2:y3+h-2,x3-1:x3+w-1];
    I1 = 255*im1.astype('double')/im1.max();
    I2 = 255*im2.astype('double')/im2.max();
    I3 = 255*im3.astype('double')/im3.max();
    RGB=np.empty([h,w,3],dtype=float)
    RGB[:,:,0]=I3
    RGB[:,:,1]=I2
    RGB[:,:,2]=I1
    plt.show()
    plt.imshow(RGB/255)
    return RGB

def cropImage(img,point,extents):
    crop=img[point[1]-extents:point[1]+extents,point[0]-extents:point[0]+extents] 
    return crop

def cropImage2(img,cropSize):
    imgAncho=img.shape[1]
    imgAlto=img.shape[0]    
    beginCorner=[0,0]
    endCorner=[imgAlto,imgAncho]
    cropImg=img[(beginCorner[0]+cropSize[2]):(endCorner[0]+cropSize[3]),(beginCorner[1]+cropSize[0]):(endCorner[1]+cropSize[1])]
    return cropImg    
    
def correlationMatrix(img1,img2):
    matrix=signal.correlate2d(img1,img2,mode="full",boundary="fill",fillvalue=3)
    return matrix    

def refreshCropSizes(cropSizes, cropSize):
    for vector in cropSizes:
        for i in range(len(cropSize)):
            if cropSize[i]>0:
                vector[i+1]=vector[i+1]-cropSize[i]
                print "vector[i+1]",vector[i+1]
            elif cropSize[i]<0:
                vector[i-1]=vector[i-1]-cropSize[i]
                print "vector[i-1]",vector[i-1]
        print "vector",vector
    print "cropSizes",cropSizes
    
def generateCropSize(despVector):
    cropSize=[0,0,0,0]
    for i in range(len(despVector)):
        if(despVector[i]>0):
            cropSize[2*i]=despVector[i]
        elif despVector[i]<0:
            cropSize[2*i+1]=despVector[i]
    return cropSize

def generateNewRGB(s,rgbCopia,diffila,difcolumna):
    rgbCopia[:,:,s]=np.roll(rgbCopia[:,:,s],diffila,axis=0)
    rgbCopia[:,:,s]=np.roll(rgbCopia[:,:,s],difcolumna,axis=1)

def alineacion(image,rgbCopia,filtro,escala):
    centerPoint=(140,165)    
    extents=15
    if escala<1:return
    cropImg=np.empty((extents*2,extents*2,3),dtype=float)   
    cropImg[:,:,0]=cropImage(image[:,:,0],centerPoint,extents)
    for i in range(image.shape[2]):
        cropImg[:,:,i]=cropImage(lowFilter(image[:,:,i],filtro),centerPoint,extents*escala)[::escala,::escala]
        cropImg[:,:,i]=cropImg[:,:,i]-np.mean(cropImg[:,:,i])
    channelR=cropImg[:,:,0]
    channelRGradient=np.array(np.gradient(channelR))
    for i in range(2):  
        plt.show()
        plt.subplot(1,2,1)
        plt.imshow(cropImg[:,:,i+1])
        plt.title("Original")
        cropGradient=np.array(np.gradient(cropImg[:,:,i+1]))
        correlation=correlationMatrix(channelRGradient[0,:,:],cropGradient[0,:,:])        
        correlation=correlation-np.mean(correlation)

        plt.subplot(1,2,2)
        plt.imshow(correlation)
        plt.title("Cross Correlation")
        
        
        position=np.where(correlation==np.amax(correlation))
        despVector=[(position[0][0]-correlation.shape[0]/2)*escala,(position[1][0]-correlation.shape[1]/2)*escala]
        
        fila=despVector[0]
        columna=despVector[1]
        
        print "Desplazamiento en la altura",fila
        print "Desplazamiento en el ancho", columna
        
        generateNewRGB(i+1,rgbCopia,fila,columna)
    rgbCopia=equalize_hist(rgbCopia)
    rgbCopia=rgbCopia*255/np.max(rgbCopia)
    return rgbCopia



def piramide(imagen,rgbCopia,filtro,escala):
    while(escala>0):
        print "------------- Escala",escala,"----------------"
        imagenAlineada=alineacion(imagen,rgbCopia,filtro,escala)
        imagen=imagenAlineada
        escala=escala/2
    imagenAlineada=equalize_hist(imagenAlineada)
    imagenAlineada=imagenAlineada*255/np.max(imagenAlineada)
    
    return imagenAlineada    


def Ej1(image):
    print "##########################################"
    print "#              Alineacion                #"
    print "##########################################"
    sigma=2    
    filtro=gaussiana(sigma)
    rgbCopia=np.copy(image)
    rgbCopiaAlineada=alineacion(image,rgbCopia,filtro,sigma)

    plt.show()
    plt.imshow(rgbCopiaAlineada.astype('uint8'))
    plt.title("Alineacion lineal")


def Ej2(image):
    print "##########################################"
    print "#        Alineacion Piramidal            #"
    print "##########################################"
    sigma=4
    filtro=gaussiana(sigma)
    rgbCopia=np.copy(image)
    rgbAlineacionPiramide=piramide(image,rgbCopia,filtro,sigma)
    
    plt.show()
    plt.imshow(rgbAlineacionPiramide.astype('uint8'))
    plt.title("Alineacion piramidal")

def main():
    img = Image.open("00029u.png")
    image=getChannel(img)
    
    Ej1(image)
        
    Ej2(image)
main()

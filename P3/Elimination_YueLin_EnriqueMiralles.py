# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 14:02:30 2015

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
from interactive_only import*


def funBacktracking(matrixM):
    m2l=np.zeros((matrixM.shape[0],2),dtype=int)
    
    minimoFila = np.min(matrixM[matrixM.shape[0]-1,:])
    pos = np.where(minimoFila==matrixM[matrixM.shape[0]-1,:])
    
    fila,columna=matrixM.shape[0]-1,pos[0][0]

    filaRecorrido=fila
    while(filaRecorrido!=-1):
        m2l[fila,0]=fila
        m2l[fila,1]=columna
        if(columna==0):
            minimoFila = min(matrixM[fila-1,columna],matrixM[fila-1,columna+1]) 
            pos = np.where(minimoFila==matrixM[fila-1,columna:columna+2]) 
            fila=fila-1
        elif(columna==(matrixM.shape[1]-1)):
            minimoFila = min(matrixM[fila-1,columna-1],matrixM[fila-1,columna])  
            pos = np.where(minimoFila==matrixM[fila-1,columna-1:]) 
        else:
            minimoFila = min(matrixM[fila-1,columna-1],matrixM[fila-1,columna],matrixM[fila-1,columna+1])
            pos = np.where(minimoFila==matrixM[fila-1,(columna-1):(columna+2)]) 
            fila,columna=fila-1,columna+(pos[0][0]-1)

        filaRecorrido-=1
    
    return m2l
    

#La funcion nos marca el rayo en la imagen
def markPath(mat, path, mark_as='red'):
    assert mark_as in ['red','green','blue','black','white']
    
    if len(mat.shape) == 2:
        mat = color.gray2rgb(mat)
    
    ret = np.zeros(mat.shape)
    ret[:,:,:] = mat[:,:,:]
    
    # Preprocess image
    if np.max(ret) < 1.1 or np.max(ret) > 256: # matrix is in float numbers
        ret -= np.min(ret)
        ret /= np.max(ret)
        ret *= 256
    
    # Determinate components
    if mark_as == 'red':
        r,g,b = 255,0,0
    elif mark_as == 'green':
        r,g,b = 0,255,0
    elif mark_as == 'blue':
        r,g,b = 0,0,255
    elif mark_as == 'white':
        r,g,b = 255,255,255
    elif mark_as == 'black':
        r,b,b = 0,0,0

    # Place R,G,B
    for i in path:
        ret[i[0],i[1],0] = r
        ret[i[0],i[1],1] = g
        ret[i[0],i[1],2] = b
    return ret.astype('uint8')    

#Nos genera indices de vector 1D a partir de indices 2D
def generateIndexes(index2d,nColumns):
    indexes=[]    
    for i in index2d:
        index=i[0]*nColumns+i[1]
        indexes.append(index)
    return indexes
    
def getValues(channel,path):
    values=[]
    for point in path:
        values.append(channel[point[0],point[1]])
    return values

#Funcion que nos reduce una imagen eliminando las lineas con bajo gradiente
def imgReduce(img,path,reduceLines):
    nChannel=img.shape[2]
    reducedImg=np.zeros((img.shape[0]-reduceLines[0],img.shape[1]-reduceLines[1],img.shape[2]),dtype=np.uint8)
    nColumns=img.shape[1]
    #reducedImg=np.array(img[:,:,i])
    indexes=generateIndexes(path,nColumns)
    for i in range(nChannel):
        reducedImg[:,:,i]=np.reshape(np.delete(img[:,:,i],indexes),(reducedImg.shape[0],reducedImg.shape[1]))
    return reducedImg 

#Funcion para calcular el gradiente de una imagen para la sintesis o reduccion
def calcGradient(img): 
    imgScaleGray = color.rgb2gray(img)
    matrix_double=np.array(imgScaleGray).astype("double")
    
    gX,gY=np.gradient(matrix_double) #imgScaleGray
    return np.abs(gX)+np.abs(gY)

#Funcion para calcular el gradiente de una imagen para la eliminacion de un objeto
def calcGradientEliminate(img,points):
    gradient=calcGradient(img)
    for i in range(points[0][1],points[1][1]-1):
        for j in range(points[0][0],points[1][0]-1):
            gradient[i,j]=-100
    return gradient

#La funcion que nos devuelve la siguiente linea que podemos 
#eliminar para la operacion de eliminar un objeto
def generateDelLinesE(img,points):
    lines=[]
    gXY=calcGradientEliminate(img,points)
    
    size_Y=np.shape(gXY)[0]
    size_X=np.shape(gXY)[1]    
    M=np.zeros([size_Y,size_X],dtype=float) #type(gXY[0,0])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if(i==0):
                M[i,j] = gXY[i,j]
            else:
                if(j >= M.shape[1]-1):
                    M[i,j] = gXY[i,j]+min(M[i-1,j-1],M[i-1,j])
                else:
                    M[i,j]=gXY[i,j]+min(M[i-1,j-1],M[i-1,j],M[i-1,j+1])
                            
        
    lines=funBacktracking(M)
    return lines

#Funcion para genera losp puntos maximo y minimo que define 
#el poligono que contiene el objeto a eliminar
def generateRectangle(tupla):
    coordX=[]
    coordY=[]
    for point in tupla:
        coordX.append(int(round(point[0])))
        coordY.append(int(round(point[1])))
    minx=min(coordX)
    miny=min(coordY)
    maxx=max(coordX)
    maxy=max(coordY)
    minPoint=[minx,miny]
    maxPoint=[maxx,maxy]
    return [minPoint,maxPoint]

#Seam carving para hacer la eliminacion de un objeto
def seamCarvingElimination(path):
    print "=========================================="
    print "          Seam Carving Elimination        "   
    print "=========================================="    
    #colorImg = mpimg.imread('iberia.jpg')
    colorImg=mpimg.imread("agbar.jpg")    
    grayImg=color.rgb2gray(colorImg)   
    ioff()
    rdi = get_mouse_click(grayImg)
    points=generateRectangle(rdi.points)
    print points
    ion()
    rango=(points[1][0]-points[0][0])+1
    for i in range(rango):
        delLines=generateDelLinesE(colorImg,points)
        figuraMarcada=markPath(colorImg, delLines, mark_as='red')
        colorImg=imgReduce(colorImg,delLines,[0,1])
        points[1][0]=points[1][0]-1
        #plt.figure(1)
        #plt.show()
        #plt.imshow(figuraMarcada)
    plt.show()
    plt.imshow(colorImg) 
    
def main():
    raiz=os.getcwd()
    plt.close("all")
    seamCarvingElimination(raiz)
main()
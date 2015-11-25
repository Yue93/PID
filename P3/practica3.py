# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:39:20 2015
@author: enrique
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

def reduccionImagen():
    raiz=os.getcwd()
    print "raiz",raiz
    
    img = mpimg.imread(raiz+"\countryside.jpg")
    print "shape", img.shape
    img3=img
    img4=img
    
    #for l in range(100):
     #   M=CalculoEnergiaM(img3)
     #   lmark=funBacktrackingAbAr(M)
     #   ion()
     #   img4=markPath(img4, lmark, mark_as='red')        
     #   img3=reducir(img3,lmark)
     #   img3=img3.astype('uint8')    
        
        
    M=CalculoEnergiaM(img3)
    mldos=funBacktrackingDrIz(M)
    print "mldos", mldos        
    ion()
    img4=markPath(img, mldos, mark_as='red')
    
    plt.figure(1)
    plt.show()
    plt.imshow(img4)
    
    #plt.figure(2)
    #plt.show()
    #plt.imshow(img3)
    



def CalculoEnergiaM(img):
    
    imgScaleGray = color.rgb2gray(img)
    matrix_double=np.array(imgScaleGray).astype("double")
    
    gX,gY=np.gradient(matrix_double)
    gXY=np.abs(gX)+np.abs(gY)
    
    print "gXY", gXY    
    
    
    size_Y=np.shape(gXY)[0]
    print "size_Y", size_Y
    size_X=np.shape(gXY)[1]    
    print "size_X", size_X
    M=np.zeros([size_Y,size_X],dtype=float)
    print "M.shape",M.shape
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if(i==0):
                M[i,j] = gXY[i,j]
            else:
                if(j >= M.shape[0]-1):
                    M[i,j] = gXY[i,j]+min(M[i-1,j-1],M[i-1,j])
                else:
                    M[i,j]=gXY[i,j]+min(M[i-1,j-1],M[i-1,j],M[i-1,j+1])
    print "M", M
    return M

    
    
    
def reducir(img1,vectorEliminacion):
    size_y= np.shape(img1)[0]
    size_x = np.shape(img1)[1]
    img2=np.empty((size_y,size_x-1,3),dtype=float)
    print "img2", img2
    print "img2.shape", img2.shape
    for f in range(img2.shape[0]):
        for c in range(img2.shape[1]):
            if(f==vectorEliminacion[f,0] & c==vectorEliminacion[f,1]):
                img2[f,c]=img1[f,c+1]
            else:
                img2[f,c]=img1[f,c]
    
    return img2        
            
def funBacktrackingDrIz(matrixM):
    print "matrixM.shape", matrixM.shape
    
    m3l=np.zeros((matrixM.shape[1],2),dtype=int)

    print "m3l", m3l.shape
    
    print "M1", matrixM.shape[0]
    
    minimoColumna = np.min(matrixM[:,matrixM.shape[1]-1])
    pos = np.where(minimoColumna==matrixM[:,matrixM.shape[1]-1])
    print "minimo", minimoColumna, "posicion", pos
    
    fila,columna=pos[0][0],matrixM.shape[1]-1
    
    print "fila,columna", fila, columna
    
    columnaRecorrido=columna
    while(columnaRecorrido!=-1):
        m3l[columna,0]=fila
        m3l[columna,1]=columna
        minimoColumna = min(matrixM[fila-1,columna-1],matrixM[fila,columna-1],matrixM[fila+1,columna-1])    
        pos = np.where(minimoColumna==matrixM[:,columna-1]) 
        fila,columna=pos[0][0],columna-1
        columnaRecorrido-=1
        
    print "m3l", m3l
    
    return m3l
    
    
def funBacktrackingAbAr(matrixM):

    print "matrixM.shape", matrixM.shape
    
    m2l=np.zeros((matrixM.shape[0],2),dtype=int)

    print "m2l", m2l.shape
    
    print "M1", matrixM.shape[0]
    
    minimoFila = np.min(matrixM[matrixM.shape[0]-1,:])
    pos = np.where(minimoFila==matrixM[matrixM.shape[0]-1,:])
    print "minimo", minimoFila, "posicion", pos
    
    fila,columna=matrixM.shape[0]-1,pos[0][0]
    
    print "dato", matrixM[fila,pos]    
    print "fila,columna", fila, columna
    filaRecorrido=fila
    while(filaRecorrido!=-1):
        m2l[fila,0]=fila
        m2l[fila,1]=columna
        minimoFila = min(matrixM[fila-1,columna-1],matrixM[fila-1,columna],matrixM[fila-1,columna+1])    
        pos = np.where(minimoFila==matrixM[fila-1,:]) 
        fila,columna=fila-1,pos[0][0]
        filaRecorrido-=1
        
    print "m2l", m2l
    
    return m2l
    
    
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
    
                
    
reduccionImagen()
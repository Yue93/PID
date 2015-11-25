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



def CalculoEnergia():
    raiz=os.getcwd()
    print "raiz",raiz
    #LOAD IMAGES
    img = mpimg.imread(raiz+"\countryside.jpg")
    print "shape", img.shape
    
    imgScaleGray = color.rgb2gray(img)
    matrix_double=np.array(imgScaleGray).astype("double")
    
    gX,gY=np.gradient(matrix_double) #imgScaleGray
    gXY=np.abs(gX)+np.abs(gY)
    
    print "gXY", gXY    
    
    
    size_Y=np.shape(gXY)[0]
    print "size_Y", size_Y
    size_X=np.shape(gXY)[1]    
    print "size_X", size_X
    M=np.zeros([size_Y,size_X],dtype=float) #type(gXY[0,0])
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
                        
            #if(j==0):
            #    M[j,i] = gXY[j,i]
            #else:
            #    if(i >= M.shape[0]-1):
            #        M[j,i] = gXY[j,i]+min(M[j-1,i-1],M[j-1,i])
            #    else:
            #        M[j,i]=gXY[j,i]+min(M[j-1,i-1],M[j-1,i],M[j-1,i+1])
            #M[i,j]=gXY[i,j]+min(M[i-1,j-1],M[i-1,j],M[i-1,j+1])        
    
    print "M", M
        
    lmark=funBacktracking(M)
    #RGB=np.empty((img.shape[0],img.shape[1],img.shape[2]),dtype=float)
    ion()
    figuraMarcada=markPath(img, lmark, mark_as='red')
    plt.figure(1)
    plt.show()
    plt.imshow(figuraMarcada)
    
    #canalesRGB[:,:,0]=img[:,:,0]
    #canalesRGB[:,:,1]=img[:,:,1]
    #canalesRGB[:,:,2]=img[:,:,2]
    
    #(gRx,gRy)=np.gradient(canalesRGB[:,:,0])
    #(gGx,gGy)=np.gradient(canalesRGB[:,:,1])
    #(gBx,gBy)=np.gradient(canalesRGB[:,:,2])
    
    #sumgRxgRy=gRx+gRy
    #sumgGxgGy=gGx+gGy
    #sumgBxgBy=gBx+gBy
    
    
    #RGB[:,:,0]=sumgRxgRy
    #RGB[:,:,1]=sumgGxgGy
    #RGB[:,:,2]=sumgBxgBy
    
    #print "RGB",RGB
    
    #M=np.empty((img.shape[0],img.shape[1],img.shape[2]),dtype=float)
    #print "M",M,"M.shape",M.shape
    #for i in range(M.shape[0]):
     #   for j in range(M.shape[1]):
      #      M(i,j)=RGB(i,j)+min(M(i-1,j-1),M(i-1,j),M(i-1,j+1))
def funBacktracking(matrixM):

    print "matrixM.shape", matrixM.shape
    #M1=np.empty((matrixM.shape[0],matrixM.shape[1]),dtype=float)
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
        #m2l[fila,2]=minimoFila
        minimoFila = min(matrixM[fila-1,columna-1],matrixM[fila-1,columna],matrixM[fila-1,columna+1])    
        pos = np.where(minimoFila==matrixM[fila-1,:]) 
        fila,columna=fila-1,pos[0][0]
        filaRecorrido-=1
        
    print "m2l", m2l
    
    return m2l
    
    #print "minimoFila", minimoFila
    #print "fila1,columna1", fila1, columna1
    
    #print "filaUltima", matrixM[matrixM.shape[0]-1,:]
    #print "MatrixM", matrixM
    #matrixfila=min(matrixM[matrixM.shape[0]-1])
    #x=0
    #print "matrixfila", matrixfila

    
#print "matrixfila", matrixfila

#while (xfin!=0 or yfin!=0):
    

#for s in M1.shape[0]:
    

#return M1
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
    
                
    
CalculoEnergia()
  
    
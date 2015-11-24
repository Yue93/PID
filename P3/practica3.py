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



def CalculoEnergia():
    raiz=os.getcwd()
    print "raiz",raiz
    #LOAD IMAGES
    img = mpimg.imread(raiz+"\countryside.jpg")
    print "shape", img.shape
    
    imgScaleGray = color.rgb2gray(img)
    matrix_double=np.array(imgScaleGray).astype("double")
    
    gX,gY=np.gradient(matrix_double) #imgScaleGray
    gXY=gX+gY
    
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
        
    funBacktracking(M)
    #RGB=np.empty((img.shape[0],img.shape[1],img.shape[2]),dtype=float)
    
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
    M1=np.empty((matrixM.shape[0],matrixM.shape[1]),dtype=float)
    print "M1", matrixM.shape[0]
    
    min = np.min(matrixM[matrixM.shape[0]-1])
    minim = np.argmin(matrixM[matrixM.shape[0]-1])
    print "minim", minim
    pos = np.where(min==matrixM[matrixM.shape[0]-1,:])
    print "minimo", min, "posicion", pos
    
    print "filaUltima", matrixM[matrixM.shape[0]-1,:]
    print "MatrixM", matrixM
    #matrixfila=min(matrixM[matrixM.shape[0]-1])
    #x=0
    #print "matrixfila", matrixfila

    
#print "matrixfila", matrixfila

#while (xfin!=0 or yfin!=0):
    

#for s in M1.shape[0]:
    

#return M1
    
    
                
    
CalculoEnergia()
  
    
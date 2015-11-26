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
from interactive_only import*

def normalize(matriz):
    minValue=matriz.min()
    maxValue=matriz.max()
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            matriz[i,j]=(matriz[i,j]-minValue)/(maxValue-minValue)



def funBacktracking(matrixM):

    #print "matrixM.shape", matrixM.shape
    #M1=np.empty((matrixM.shape[0],matrixM.shape[1]),dtype=float)
    m2l=np.zeros((matrixM.shape[0],2),dtype=int)

    #print "m2l", m2l.shape
    
    #print "M1", matrixM.shape[0]
    
    minimoFila = np.min(matrixM[matrixM.shape[0]-1,:])
    pos = np.where(minimoFila==matrixM[matrixM.shape[0]-1,:])
    #print "minimo", minimoFila, "posicion", pos
    
    fila,columna=matrixM.shape[0]-1,pos[0][0]
    
    #print "dato", matrixM[fila,pos]    
    #print "fila,columna", fila, columna
    filaRecorrido=fila
    while(filaRecorrido!=-1):
        m2l[fila,0]=fila
        m2l[fila,1]=columna
        #m2l[fila,2]=minimoFila
        if(columna==0):
            #print "==========Start=========="
            minimoFila = min(matrixM[fila-1,columna],matrixM[fila-1,columna+1]) 
            pos = np.where(minimoFila==matrixM[fila-1,columna:columna+2]) 
            fila=fila-1
        elif(columna==(matrixM.shape[1]-1)):
            #print "==========Final=========="            
            minimoFila = min(matrixM[fila-1,columna-1],matrixM[fila-1,columna])  
            pos = np.where(minimoFila==matrixM[fila-1,columna-1:]) 
        else:
            #print "==========Medio=========="
            #print "Fila, columna",fila,columna
            #print "Fila",matrixM[fila-1,(columna-1):(columna+2)]
            minimoFila = min(matrixM[fila-1,columna-1],matrixM[fila-1,columna],matrixM[fila-1,columna+1])
            pos = np.where(minimoFila==matrixM[fila-1,(columna-1):(columna+2)]) 
            fila,columna=fila-1,columna+(pos[0][0]-1)
        #print "----MinimoFila",minimoFila
        #print "==========Pos==============",pos
        #fila,columna=fila-1,columna+(pos[0][0]-1)
        #print "Fila, Columna",fila, columna
        filaRecorrido-=1
        
    #○print "m2l", m2l
    
    return m2l
    

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
    
def imgReduce(img,path,reduceLines):
    nChannel=img.shape[2]
    reducedImg=np.zeros((img.shape[0]-reduceLines[0],img.shape[1]-reduceLines[1],img.shape[2]),dtype=np.uint8)
    nColumns=img.shape[1]
    #reducedImg=np.array(img[:,:,i])
    indexes=generateIndexes(path,nColumns)
    for i in range(nChannel):
        reducedImg[:,:,i]=np.reshape(np.delete(img[:,:,i],indexes),(reducedImg.shape[0],reducedImg.shape[1]))
    return reducedImg 
    
def imgExtend(img,path,extendLines):
    nChannel=img.shape[2]
    nColumns=img.shape[1]
    extendedImg=np.zeros((img.shape[0]+extendLines[0],img.shape[1]+extendLines[1],img.shape[2]),dtype=np.uint8)
    #reducedImg=np.array(img[:,:,i])
    indexes=generateIndexes(path,nColumns)
    for i in range(nChannel):
        values=getValues(img[:,:,i],path)
        extendedImg[:,:,i]=np.reshape(np.insert(img[:,:,i],indexes,values),(extendedImg.shape[0],extendedImg.shape[1]))
    return extendedImg
  
def calcGradient(img): 
    imgScaleGray = color.rgb2gray(img)
    matrix_double=np.array(imgScaleGray).astype("double")
    
    gX,gY=np.gradient(matrix_double) #imgScaleGray
    return np.abs(gX)+np.abs(gY)

def calcGradientEliminate(img,points):
    gradient=calcGradient(img)
    for i in range(points[0][1]-1,points[1][1]):
        for j in range(points[0][0]-1,points[1][0]):
            gradient[i,j]=-100
    return gradient
 
def generateDelLines(img):
    #print "shape", img.shape
    #nReduction=50
    #for i in range(nReduction):
    lines=[]
    gXY=calcGradient(img)
    #print "gXY", gXY    
    
    size_Y=np.shape(gXY)[0]
    #print "size_Y", size_Y
    size_X=np.shape(gXY)[1]    
    #print "size_X", size_X
    M=np.zeros([size_Y,size_X],dtype=float) #type(gXY[0,0])
    #print "M.shape",M.shape
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
    #RGB=np.empty((img.shape[0],img.shape[1],img.shape[2]),dtype=float)
    return lines

def generateDelLinesE(img,points):
    #print "shape", img.shape
    #nReduction=50
    #for i in range(nReduction):
    lines=[]
    gXY=calcGradientEliminate(img,points)
    #print "gXY", gXY    
    
    size_Y=np.shape(gXY)[0]
    #print "size_Y", size_Y
    size_X=np.shape(gXY)[1]    
    #print "size_X", size_X
    M=np.zeros([size_Y,size_X],dtype=float) #type(gXY[0,0])
    #print "M.shape",M.shape
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
    #RGB=np.empty((img.shape[0],img.shape[1],img.shape[2]),dtype=float)
    return lines

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
    
    

def seamCarvingReduction(path):
    print "=========================================="
    print "           Seam Carving Reduction         "   
    print "=========================================="    
    img = mpimg.imread(path+"\\towelsmall.jpg")
    #img = mpimg.imread(path+"\iberia.jpg")
    reduceSize=[0,50]
    
    for nlines in reduceSize:
        for i in range(nlines):
            delLines=generateDelLines(img)
            figuraMarcada=markPath(img, delLines, mark_as='red')
            img=imgReduce(img,delLines,[0,1])
            plt.show()
            plt.imshow(figuraMarcada)
            #ion()
            #plt.figure(1)
            #plt.show()
            #plt.imshow(figuraMarcada)
    #newImg=imgReduce(img,lmark)
    #colorImg=color.gray2rgb(newImg)
    #print "Final image shape", img.shape
    plt.show()
    plt.imshow(img)    
    

    
def seamCarvingElimination(path):
    print "=========================================="
    print "          Seam Carving Elimination        "   
    print "=========================================="    
    colorImg = mpimg.imread('iberia.jpg')
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
        #plt.figure(i)
        plt.show()
        plt.imshow(figuraMarcada)
        #ion()
        #plt.figure(i)
        #plt.show()
        #plt.imshow(figuraMarcada)
    #newImg=imgReduce(img,lmark)
    #colorImg=color.gray2rgb(newImg)
    #print "Final image shape", img.shape
    #ion()
    #plt.figure(1)
    #plt.figure()
    print colorImg.shape
    plt.show()
    plt.imshow(colorImg)  
    
def seamCarvingSintesis(path):
    print "=========================================="
    print "           Seam Carving Syntesis         "   
    print "=========================================="    
    #img = mpimg.imread(path+"\countryside.jpg")
    #img = mpimg.imread("iberia.jpg")    
    img = mpimg.imread("towelsmall.jpg")    
    reduceSize=[0,50]
    
    plt.show()
    plt.imshow(img)
    for nlines in reduceSize:
        for i in range(nlines):
            duplicateLines=generateDelLines(img)
            figuraMarcada=markPath(img, duplicateLines, mark_as='red')
            img=imgExtend(img,duplicateLines,[0,1])
            #plt.show()
            #plt.imshow(figuraMarcada)
            #plt.show()
            #plt.imshow(figuraMarcada)
    #newImg=imgReduce(img,lmark)
    #colorImg=color.gray2rgb(newImg)
    #print "Final image shape", img.shape
    plt.show()
    plt.imshow(img)  
    
def main():
    raiz=os.getcwd()
    plt.close("all")
    #seamCarvingReduction(raiz)
    #seamCarvingElimination(raiz)    
    seamCarvingSintesis(raiz)
main()
    
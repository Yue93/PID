# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:06:32 2015

@author: enrique
"""


def gaussiana(oscRange,sigma):
    #oscRange=6*sigma/2
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


def TDFourier():
    
    raiz=os.getcwd()
    im = io.imread(raiz+"\human.png")
    fftsize=1024
    im_fft = fftpack.fft2(im, (fftsize, fftsize))
    hs = 50;
    filgaussiano=gaussiana(hs*2+1,9)
    fil_fft = fft2(filgaussiano, fftsize, fftsize)
    im_fil_fft = im_fft * fil_fft
    im_fil = ifft2(im_fil_fft)
    im_fil = im_fil(1+hs:size(im,1)+hs,1+hs:size(im, 2)+hs)
    
    
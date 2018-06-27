import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
img = cv.imread('/home/socmgr/barack.jpeg')

kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
#kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
#kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
len(kernel)
pad = int((len(kernel) - 1)/2)
stride = 1
def convolve(image, kernel, padding, stride):
    (ih,iw,ch) = image.shape[:3]
    (kh,kw) = kernel.shape[:2]
    oh = int((ih+2*padding-kh)/stride)+1
    ow = int((iw+2*padding-kw)/stride)+1
    output = np.zeros((oh,ow,ch),dtype="float32")
    
    for y in np.arange(0,oh):
        for x in np.arange(0,ow):
            for c in np.arange(0,ch):
                result = 0
                for h in np.arange(0,kh):
                    for w in np.arange(0,kw):
                        W = x*stride + w
                        H = y*stride + h
                        
                        if W<iw and W>0 and H<ih and H>0:
                            result = image[H,W,c]*kernel[h,w] + result
                            
                #if result > 255:
                #    result = result-255
                #elif result < 0:
                #    result = result+255
                
                output[y,x,c] = result
                
                if output[y,x,c] > 256:
                    output[y,x,c] = 255
                elif output[y,x,c] < 0:
                    output [y,x,c] = 0
                
    
    #print(output[100][100])
    #output = rescale_intensity(output, in_range=(0, 255))
    #print(output[100][100])
    #output = (output*255).astype("uint8")
    #print(output[100][100])
                
                
    
    return output
    
conv = convolve(img,kernel,pad,stride)
conv = conv.astype("uint8")

cv.imshow('conv',conv)
cv.waitKey(0)
cv.destroyAllWindows()

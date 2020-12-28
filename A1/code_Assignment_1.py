from PIL import Image
import numpy as np
import math
from scipy import signal
import cv2
import os
import time

# Opens image with name image_name using PIL 
# Returns PIL image
def openImage(image_name):
    script_dir = os.path.dirname(__file__)
    image_path = script_dir + image_name
    image = Image.open(image_path)
    return image

# Creates PIL image from array and returns
# Array is converted to unsigned integer format
def createImage(array):
    return Image.fromarray(array.astype('uint8'))


# Returns a box filter of size n by n
# Sum of all must be 1
#N must be odd -> if not error
def boxfileter(n):
    #if n is even, assert
    assert n % 2 != 0
    #Sum of all array must be 1.
    s = 1 / (math.pow(n, 2)) 

    #Create n by n array 
    ret = np.full((n, n), s)

    return ret

# Caluclates Guassian of X
def gaussianf(x, sigma):
    return (math.exp((-1 * (math.pow(x, 2))) / (2 * (math.pow(sigma, 2)))))

# Returns 1D Gaussian filter fo sigma
def gauss1d(sigma):
    
    #Size of the return Array
    n = math.ceil(sigma * 6)

    #Want an off number  
    if(n % 2 == 0):
        n = n + 1

    #Create array values from -n/2 to n/2
    startVal = ( -1 * (math.floor(n / 2)))
    stopVal = math.floor(n /2) + 1

    values = np.arange(start = startVal , stop =stopVal)

    
    #map array values to gaussian function
    ret = np.array([gaussianf(xi, sigma) for xi in values])


    #Normalize array
    ret /= np.sum(ret)

    return ret

# Returns 2D Gaussian filer for sigma
def gauss2d(sigma):
    #get initial 1D array
    guassian1D = gauss1d(sigma)

    #convert 1D array to 2D
    guassian2D = guassian1D[np.newaxis]

    #find transpose
    transpose = np.transpose(guassian2D)

    #get the convolution of the 2D guassian with its transpose
    guassianRet = signal.convolve2d(guassian2D, transpose)
    return guassianRet

#Helper function for padding
#References: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


# Takes an image (array) and a filter, and perfors convolution to the image with zere paddigns
# Both imput variables are in type 'np.float32'
#Convolution formula: I'(X, Y) = sum_j={-1} ^ {1} sum_i={-1} ^ 1 (F(-i, -j) * I(X + i, Y + j)) 
def covolve2d_manual(array, filter):

    # Size of image
    row_image = array.shape[0]
    column_image = array.shape[1]

    # Size of filter
    row_filter = filter.shape[0]
    column_filter = filter.shape[1]

    # How much padding are we needing
    row_extra = math.floor(row_filter / 2)
    column_extra =  math.floor(column_filter / 2)

    #  Initialize return array of same size as image but all zeros
    ret = np.zeros_like(array)

    #Create copy image with extra columns and rows on the side 
    array_side = np.pad(array, (row_extra, column_extra), pad_with)
    
    # Itearte througth the image
    for x in range(row_image):
        for y in range(column_image):
            #Find new value by multiplying filter with neighbour of value on padded image
            ret[x, y] = (filter * array_side[x: x + row_filter, y: y + column_filter]).sum()
   
    return ret


# Applies Gaussian convolution to a 2D array for the value of sigma
def gaussconvolve2d_manual(array,sigma):
    filter = gauss2d(sigma)
    ret = covolve2d_manual(array, filter)
    return ret

def gaussconvolve2d_scipy(array,sigma):
    filter = gauss2d(sigma)
    ret = signal.convolve2d(array, filter, 'same')
    return ret


# Apply Gaussian convolution to all colour channels and return filtered array.
def filterAllColourChanels(array, sigma):
    filteredArray = [gaussconvolve2d_manual(array[:, :, c], sigma) for c in range(len(array[0][0]))]
    return np.stack(filteredArray, axis = 2)


def highFrequencyImage(array, sigma):
    filteredArray = filterAllColourChanels(array, sigma)
    return (array - filteredArray)
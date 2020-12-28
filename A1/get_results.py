import code_Assignment_1 as sol
import sys
from PIL import Image
import numpy as np
import math
from scipy import signal
import cv2
import os
import time
from matplotlib import pyplot as plt

def main():
    # Part 2.1
    # print(sol.boxfileter(3))
    # try:
    #     print(sol.boxfileter(4))

    # except AssertionError:
    #   print("Assertion error expected")
    # print(sol.boxfileter(5))


    # Part 2.2
    #  print( sol.gauss1d(0.3))
    #  print( sol.gauss1d(0.5))
    #  print( sol.gauss1d(1))
    #  print( sol.gauss1d(2))

    # Part 2.3
    # print(sol.gauss2d(0.5))
    # print(sol.gauss2d(1))

    # Part 2.4
    # Open image dog and filter using sigmna 3
    #  image = sol.openImage('/0b_dog.bmp')
    #  image = image.convert('L')
    #  image.show()
    #  data = np.array(image, dtype = np.float32)
    #  converted_array = sol.gaussconvolve2d_manual(data, 3)
    #  converted_image = sol.createImage(converted_array)
    #  converted_image.show()

    # Part 2.5
    # Open image dog and filter using sigmna 3
    # image = sol.openImage('/0b_dog.bmp')
    # image = image.convert('L')
    # image.show()
    # data = np.array(image, dtype = np.float32)
    # converted_array = sol.gaussconvolve2d_scipy(data, 3)
    # converted_image = sol.createImage(converted_array)
    # converted_image.show()

    #Part 2.6
    #compare using time
    # image = sol.openImage('/dog.jpg')
    # image = image.convert('L')
    # data = np.array(image, dtype = np.float32)
    # t1 = time.time()
    # converted_array_mine = sol.gaussconvolve2d_manual(data, 10)
    # print("Duration of mine: ", time.time() - t1)

    # t2 = time.time()
    # converted_array_scipy = sol.gaussconvolve2d_scipy(data, 10)
    # print("Duration of scipy: ", time.time() - t2)

    # converted_image_mine = sol.createImage(converted_array_mine)
    # converted_image_scipy = sol.createImage(converted_array_scipy)


    #Part 2.7
    #
    


    # Part 3.1
    # Open image dog and filter using sigmna 3
    # image = sol.openImage('/dog.jpg')
    # image.show()
    # data = np.array(image, dtype = np.float32)
    # converted_array = sol.filterAllColourChanels(data, 10)
    # converted_image = sol.createImage(converted_array)
    # converted_image.show()

    # Part 3.2
    # image = sol.openImage('/0a_cat.bmp')
    # data = np.array(image, dtype = np.float32)
    # converted_array = sol.highFrequencyImage(data, 10)
    # converted_image = sol.createImage(converted_array + 128)
    # converted_image.show()

    # Part 3.3
    # image_one = sol.openImage('/1a_bicycle.bmp')
    # image_two = sol.openImage('/1b_motorcycle.bmp')
    # data_one = np.array(image_one, dtype = np.float32)
    # data_two = np.array(image_two, dtype = np.float32)
    
    # converted_array_high_one = sol.highFrequencyImage(data_one, 5)
    # converted_array_low_one = sol.filterAllColourChanels(data_two, 5)
    # converted_array_total_one = converted_array_high_one + converted_array_low_one
    # converted_image_one = sol.createImage(converted_array_total_one)
    # converted_image_one.show()

    # converted_array_high_two = sol.highFrequencyImage(data_one, 10)
    # converted_array_low_two = sol.filterAllColourChanels(data_two, 10)
    # converted_array_total_two = converted_array_high_two + converted_array_low_two
    # converted_image_two = sol.createImage(converted_array_total_two)
    # converted_image_two.show()

    # converted_array_high_three = sol.highFrequencyImage(data_one, 15)
    # converted_array_low_three = sol.filterAllColourChanels(data_two, 15)
    # converted_array_total_three = converted_array_high_three + converted_array_low_three
    # converted_image_three = sol.createImage(converted_array_total_three)
    # converted_image_three.show()
    
    # Part 4.1
    # image_one = cv2.imread("c:/Users/tranc/OneDrive/Escritorio/clase/CPSC425/A1/box_gauss.png")
    # image_two = cv2.imread("c:/Users/tranc/OneDrive/Escritorio/clase/CPSC425/A1/box_speckle.png")

    # image_blur_1 = cv2.GaussianBlur(image_one, ksize = (7, 7), sigmaX = 200) 
    # image_blur_2 = cv2.GaussianBlur(image_two, ksize = (7, 7), sigmaX = 100)

    # image_bil_1 = cv2.bilateralFilter(image_one, d = 30, sigmaColor = 200, sigmaSpace = 20)
    # image_bil_2 = cv2.bilateralFilter(image_two, d = 30, sigmaColor = 200, sigmaSpace = 200)

    # image_Medblur_1 = cv2.medianBlur(image_one, ksize = 5)
    # image_Medblur_2 = cv2.medianBlur(image_two, ksize = 11)


    # plt.imshow(image_Medblur_2)
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

    # Part 4.2
    # image_one = cv2.imread("c:/Users/tranc/OneDrive/Escritorio/clase/CPSC425/A1/box_gauss.png")
    # image_two = cv2.imread("c:/Users/tranc/OneDrive/Escritorio/clase/CPSC425/A1/box_speckle.png")

    # image_one = cv2.GaussianBlur(image_one, ksize = (7, 7), sigmaX = 50)
    # image_one = cv2.bilateralFilter(image_one, 7, sigmaColor = 150, sigmaSpace = 150)
    # image_one = cv2.medianBlur(image_one, 7)

    # image_two = cv2.GaussianBlur(image_two, ksize = (7, 7), sigmaX = 50)
    # image_two = cv2.bilateralFilter(image_two, 7, sigmaColor = 150, sigmaSpace = 150)
    # image_two = cv2.medianBlur(image_two, 7)

    # plt.imshow(image_one)
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

    # plt.imshow(image_two)
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

if __name__ == "__main__":
    main()
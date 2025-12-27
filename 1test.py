import cv2 as cv
import numpy as np
imgpath = r'dracma.jpg'
imgresultpath = r'result-test.jpg'
#----------------------------------------------------------
im = cv.imread(imgpath)
im_height, im_width, im_channels = im.shape
#----------------------------------------------------------
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#----------------------------------------------------------
imgray_bit_xor = cv.bitwise_not(imgray)
#----------------------------------------------------------
#do canny:
thresholdCanny = 13
#thresholdCanny = 17
#thresholdCanny = 8
#thresholdCanny = 52
edges = cv.Canny(imgray_bit_xor,thresholdCanny,thresholdCanny*3,3)
#----------------------------------------------------------
#do dilate:
kernel_dilate = np.ones((im_height,im_width,3), np.uint8)
kernel_dilate_blank = kernel_dilate[0:im_height,0:im_width] = (255,255,255)
dilatation_dst = cv.dilate(edges, kernel_dilate_blank, iterations = 1 )
#----------------------------------------------------------
#find contours:
contours_internal, hierarchy = cv.findContours(dilatation_dst, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
#Crea img en blanco:
blank_image = np.zeros((im_height,im_width,3), np.uint8)
blank_image[0:im_height,0:im_width] = (255,255,255) 
black_image = np.zeros((im_height,im_width,3), np.uint8)
#black_image[0:im_height,0:im_width] = (255,255,255) 
#img_with_contours = cv.drawContours(blank_image, contours_internal, 3, (0,255,0), 3)
#----------------------------------------------------------
img_with_contours = im
for contour in contours_internal:
    print(cv.contourArea(contour))
    #img_with_contours = cv.drawContours(blank_image, contour, 3, (0,255,0), 3)
    #img_with_contours = cv.drawContours(imgray_bit_xor, contour, 3, (0,255,0), 3)
    #img_with_contours = cv.drawContours(imgray, contour, 3, (0,255,0), 3)
    img_with_contours = cv.drawContours(img_with_contours, contour, 3, (0,255,0), 3)
    #imResize = cv.resize(img_with_contours, (960, 540)) 
    #cv.imshow('Contours', imResize) 
    #cv.waitKey(0)
    #cv.destroyAllWindows()

#imResize = cv.resize(img_with_contours, (960, 540)) 
#cv.imshow('Contours', imResize) 
#cv.waitKey(0)
#cv.destroyAllWindows()
imResize = cv.resize(edges, (960, 540)) 
cv.imshow('Contours', imResize) 
cv.waitKey(0)
cv.destroyAllWindows()

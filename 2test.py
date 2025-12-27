import cv2 as cv
import numpy as np
#imgpath = r'E:\win64\nodeproyects\videosinteractivosjs\server\opencv\0420.png'
imgpath = r'E:\win64\nodeproyects\videosinteractivosjs\server\opencv\196al110.jpg'

imgresultpath = r'E:\win64\nodeproyects\videosinteractivosjs\server\opencv\result-test.png'
#----------------------------------------------------------
im = cv.imread(imgpath)
im_height, im_width, im_channels = im.shape
#----------------------------------------------------------
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#----------------------------------------------------------
imgray_bit_xor = cv.bitwise_not(imgray)
#----------------------------------------------------------
#do canny:
thresholdCanny = 150
edges = cv.Canny(imgray_bit_xor,thresholdCanny,thresholdCanny*3,3)
#----------------------------------------------------------
#xor_version:
edges_xor = cv.bitwise_not(edges)

kernel_dilate = np.ones((im_height,im_width,3), np.uint8)
kernel_dilate_blank = kernel_dilate[0:im_height,0:im_width] = (255,255,255)
dilatation_dst = cv.dilate(edges, kernel_dilate_blank, iterations = 1 )

#find contours:
contours_internal, hierarchy = cv.findContours(dilatation_dst, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

#do dilate:
kernel_dilate_xor = np.ones((im_height,im_width,3), np.uint8)
kernel_dilate_blank_xor = kernel_dilate_xor[0:im_height,0:im_width] = (255,255,255)
dilatation_dst_xor = cv.dilate(edges_xor, kernel_dilate_blank_xor, iterations = 1 )
#----------------------------------------------------------
#find contours:
contours_internal_xor, hierarchy_xor = cv.findContours(dilatation_dst_xor, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

#Crea img en blanco:
##blank_image = np.zeros((im_height,im_width,3), np.uint8)
##blank_image[0:im_height,0:im_width] = (255,255,255) 
##black_image = np.zeros((im_height,im_width,3), np.uint8)
#black_image[0:im_height,0:im_width] = (255,255,255) 
#img_with_contours = cv.drawContours(blank_image, contours_internal, 3, (0,255,0), 3)
#----------------------------------------------------------
img_with_contours = im
img_with_contours_xor = im

print("start contours:")
for contour in contours_internal:
    print(cv.contourArea(contour))
    img_with_contours = cv.drawContours(img_with_contours, contour, 3, (0,255,0), 3)

print("start contours_xor:")
for contour_xor in contours_internal_xor:
    print(cv.contourArea(contour_xor))
    img_with_contours_xor = cv.drawContours(img_with_contours_xor, contour_xor, 3, (0,255,0), 3)

h = int(960/1.3)
w = int(540/1.3)
imResize = cv.resize(img_with_contours, (h, w)) 
imResize_xor = cv.resize(img_with_contours_xor, ( h, w )) 

cv.imshow('Contornos', imResize) 

cv.imshow('Contornos_xor', imResize_xor)
cv.waitKey(0)
cv.destroyAllWindows()
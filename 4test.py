import cv2 as cv
import numpy as np
import time
video1path = r'E:\win64\nodeproyects\videosinteractivosjs\server\opencv\video1.mp4'
vidcap = cv.VideoCapture( video1path )
success,im = vidcap.read()
count = 0
framepath = ''
todoslosframes = []
alpha = 1.4 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)
position = (10,50)
while success:
  success,im = vidcap.read()
  im_contrast = cv.convertScaleAbs(im, alpha=alpha, beta=beta)
  im_height, im_width, im_channels = im.shape
  #----------------------------------------------------------
  imgray = cv.cvtColor(im_contrast, cv.COLOR_BGR2GRAY)
  #----------------------------------------------------------
  imgray_bit_xor = cv.bitwise_not(imgray)
  #----------------------------------------------------------
  #do canny:
  thresholdCanny = 25#30 52 150 8 13 17
  edges = cv.Canny(imgray_bit_xor,thresholdCanny,thresholdCanny*3,3)
  kernel_dilate = np.ones((im_height,im_width,3), np.uint8)
  kernel_dilate_blank = kernel_dilate[0:im_height,0:im_width] = (255,255,255)
  dilatation_dst = cv.dilate(edges, kernel_dilate_blank, iterations = 1 )
  #----------------------------------------------------------
  #find contours:
  contours_internal, hierarchy = cv.findContours(dilatation_dst, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
  img_with_contours = im
  print("start contours:")
  for contour in contours_internal:
    print(cv.contourArea(contour))
    if cv.contourArea(contour) == 0.0:
      break
    img_with_contours = cv.drawContours(img_with_contours, contour, 3, (0,255,0), 3)
  
  cv.putText(img_with_contours,str(count),position,cv.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)#numpy array on which text is written,text,position at which writing has to start,font family,font size,font color,font stroke
  #todoslosframes[count] = img_with_contours
  todoslosframes.append(img_with_contours)
  if count >= 100:
    framepath = r'E:\win64\nodeproyects\videosinteractivosjs\server\opencv\frame'+str(count)+'.jpg'
    cv.imwrite(framepath, im)     # save frame as JPEG file
    break
  count += 1
  #----------------------------------------------------------     
cv.namedWindow('Contornos', cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty('Contornos',cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
for frame in todoslosframes:
  cv.imshow('Contornos', frame)
  time.sleep(0.03)#24frames
  if cv.waitKey(10) == 27:# exit if Escape is hit
       cv.destroyAllWindows()
       break
  
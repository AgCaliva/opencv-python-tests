import cv2 as cv
import numpy as np
video1path = r'E:\win64\nodeproyects\videosinteractivosjs\server\opencv\video1.mp4'
vidcap = cv.VideoCapture( video1path )
success,im = vidcap.read()
count = 0
framepath = ''
while success:
  success,im = vidcap.read()
  im_height, im_width, im_channels = im.shape
  #----------------------------------------------------------
  imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
  #----------------------------------------------------------
  imgray_bit_xor = cv.bitwise_not(imgray)
  #----------------------------------------------------------
  #do canny:
  thresholdCanny = 30
  edges = cv.Canny(imgray_bit_xor,thresholdCanny,thresholdCanny*3,3)
  kernel_dilate = np.ones((im_height,im_width,3), np.uint8)
  kernel_dilate_blank = kernel_dilate[0:im_height,0:im_width] = (255,255,255)
  dilatation_dst = cv.dilate(edges, kernel_dilate_blank, iterations = 1 )
  #find contours:
  contours_internal, hierarchy = cv.findContours(dilatation_dst, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
  img_with_contours = im
  print("start contours:")
  for contour in contours_internal:
    print(cv.contourArea(contour))
    
    img_with_contours = cv.drawContours(img_with_contours, contour, 3, (0,255,0), 3)
    position = (10,50)
    cv.putText(
     img_with_contours, #numpy array on which text is written
     str(count), #text
     position, #position at which writing has to start
     cv.FONT_HERSHEY_SIMPLEX, #font family
     1, #font size
     (209, 80, 0, 255), #font color
     3) #font stroke

  cv.namedWindow('Contornos', cv.WND_PROP_FULLSCREEN)
  cv.setWindowProperty('Contornos',cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
  cv.imshow('Contornos', img_with_contours) 

  if cv.waitKey(10) == 27:# exit if Escape is hit
       cv.destroyAllWindows()
       break
  if count >= 150:
      framepath = r'E:\win64\nodeproyects\videosinteractivosjs\server\opencv\frame'+str(count)+'.jpg'
      cv.imwrite(framepath, im)     # save frame as JPEG file
      break
  count += 1
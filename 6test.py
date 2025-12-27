import cv2 as cv
import numpy as np
import time
pasoporpaso = True
def printMenu():
  print('''Menu:
  N.Siguiente
  X.Aumenta numero de frames que se pasan.
  F.Buscar un frame en especifico
  0.Exit''')
def showDebug(title,im):
    cv.imshow(title, im)
    printMenu()
framepath = ''
todoslosframes = []
alpha = 1 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)
position = (10,50)
kernel_morphgrad = np.ones((2,2),np.uint8)
thresholdCanny = 20#30 52 150 8 13 17
img_with_contours = None 
count = 0
while_needcontinue = False

def algo1(img):
  global count,img_with_contours,while_needcontinue
  if pasoporpaso:
    showDebug('Imagen Original', img)
    key = cv.waitKey(0)
    if key == 48:#0
      print("press 0")
      cv.destroyAllWindows()
    elif (key == 78 or key == 110 ):#N o n
      print("press N")
      cv.destroyAllWindows()
      count += 1
      while_needcontinue = True
      return
    elif (key == 70 or key == 102):
      print("press F")
      pressItems = ""
      while True:
        press = cv.waitKey(0)
        print("press: "+chr(press))
        if press == 13:
          break
        pressItems += chr(press)
      print("pressItems: "+pressItems)
      count = int(pressItems)
      while_needcontinue = True
      return
  im_contrast = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
  if pasoporpaso:
    showDebug('ContrasteYBrillo', im_contrast) 
    key = cv.waitKey(0)
    if key == 48:#0
      cv.destroyAllWindows()
    elif (key == 78 or key == 110 ):#N o n
      cv.destroyAllWindows()
      count += 1
      while_needcontinue = True
      return
  im_height, im_width, im_channels = img.shape
  #----------------------------------------------------------
  imgray = cv.cvtColor(im_contrast, cv.COLOR_BGR2GRAY)
  if pasoporpaso:
    showDebug('Grises', imgray) 
    key = cv.waitKey(0)
    if key == 48:#0
      cv.destroyAllWindows()
    elif (key == 78 or key == 110 ):#N o n
      cv.destroyAllWindows()
      count += 1
      while_needcontinue = True
      return
  #----------------------------------------------------------
  imgray_bit_xor = cv.bitwise_not(imgray)
  if pasoporpaso:
    showDebug('Grises_xor', imgray_bit_xor) 
    key = cv.waitKey(0)
    if key == 48:#0
      cv.destroyAllWindows()
    elif (key == 78 or key == 110 ):#N o n
      cv.destroyAllWindows()
      count += 1
      while_needcontinue = True
      return
  #----------------------------------------------------------
  #do canny:
  edges = cv.Canny(imgray_bit_xor,thresholdCanny,thresholdCanny*3,3)
  if pasoporpaso:
    showDebug('Bordes', edges) 
    key = cv.waitKey(0)
    if key == 48:#0
      cv.destroyAllWindows()
    elif (key == 78 or key == 110 ):#N o n
      cv.destroyAllWindows()
      count += 1
      while_needcontinue = True
      return
  kernel_dilate = np.ones((im_height,im_width,3), np.uint8)
  kernel_dilate_blank = kernel_dilate[0:im_height,0:im_width] = (255,255,255)
  dilatation_dst = cv.dilate(edges, kernel_dilate_blank, iterations = 3 )
  if pasoporpaso:
    showDebug('Dilatacion', dilatation_dst) 
    key = cv.waitKey(0)
    if key == 48:#0
      cv.destroyAllWindows()
    elif (key == 78 or key == 110 ):#N o n
      cv.destroyAllWindows()
      count += 1
      while_needcontinue = True
      return
  #kernel_morphgrad = np.ones((3,3),np.uint8)
  img_morphgrad = cv.morphologyEx(dilatation_dst, cv.MORPH_GRADIENT, kernel_morphgrad)
  if pasoporpaso:
    showDebug('MORPH_GRADIENT', img_morphgrad) 
    key = cv.waitKey(0)
    if key == 48:#0
      cv.destroyAllWindows()
    elif (key == 78 or key == 110 ):#N o n
      cv.destroyAllWindows()
      count += 1
      while_needcontinue = True
      return
  #img_morphgrad = cv.morphologyEx(img_morphgrad, cv.MORPH_GRADIENT, kernel_morphgrad)
  img_morphgrad = cv.morphologyEx(img_morphgrad, cv.MORPH_CLOSE, kernel_morphgrad)
  if pasoporpaso:
    showDebug('MORPH_CLOSE', img_morphgrad) 
    key = cv.waitKey(0)
    if key == 48:#0
      cv.destroyAllWindows()
    elif (key == 78 or key == 110 ):#N o n
      cv.destroyAllWindows()
      count += 1
      while_needcontinue = True
      return
  img_morphgrad = cv.morphologyEx(img_morphgrad, cv.MORPH_GRADIENT, kernel_morphgrad)
  if pasoporpaso:
    showDebug('MORPH_GRADIENT', img_morphgrad) 
    key = cv.waitKey(0)
    if key == 48:#0
      cv.destroyAllWindows()
    elif (key == 78 or key == 110 ):#N o n
      cv.destroyAllWindows()
      count += 1
      while_needcontinue = True
      return
  #----------------------------------------------------------
  #find contours:
  contours_internal, hierarchy = cv.findContours(img_morphgrad, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
  img_with_contours = img
  print("start contours:")
  if len(contours_internal) > 0:
    for contour in contours_internal:
      print(cv.contourArea(contour))
      if cv.contourArea(contour) <= 0.0:
        continue
        #return
      img_with_contours = cv.drawContours(img_with_contours, contour, 3, (0,255,0), 3)
  cv.putText(img_with_contours,str(count),position,cv.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)#numpy array on which text is written,text,position at which writing has to start,font family,font size,font color,font stroke  

def showTodosLosFrames():
  global todoslosframes
  for frame in todoslosframes:
    showDebug('Contornos', frame)
    time.sleep(0.03)#24frames


def readVideo():
  global count,img_with_contours,todoslosframes,while_needcontinue
  video1path = r'E:\win64\nodeproyects\videosinteractivosjs\server\opencv\video1.mp4'
  vidcap = cv.VideoCapture( video1path )
  #otro metodo con tresh!:
  success = True
  #vidcap.set(cv.CAP_PROP_POS_FRAMES,count)
  success,im = vidcap.read()
  while success:
    cv.destroyAllWindows()
    print("count: "+str(count))
    vidcap.set(cv.CAP_PROP_POS_FRAMES,count)
    success,im = vidcap.read()
    algo1(im)
    if while_needcontinue:
      while_needcontinue = False
      continue
    todoslosframes.append(img_with_contours)
    if pasoporpaso:
      showDebug('Imagen Original + Detectados', img_with_contours) 
      key = cv.waitKey(0)
      if key == 48:#0
        cv.destroyAllWindows()
      elif (key == 78 or key == 110 ):#N o n
        cv.destroyAllWindows()
        count += 1
        continue
    #if count > 600:
     # framepath = r'E:\win64\nodeproyects\videosinteractivosjs\server\opencv\frame'+str(count)+'.jpg'
      #cv.imwrite(framepath, img_with_contours)     # save frame as JPEG file
      #break
    count += 1
    #----------------------------------------------------------     
  showTodosLosFrames()
readVideo()
  
import cv2 as cv
import numpy as np
import time

import os
import re
import multiprocessing
import time
class Scanner:
  def __init__(self):
    self.cuak = ""
    self.pasoporpaso = False
    self.testpath = ''
    self.alpha = 1 # Contrast control (1.0-3.0)
    self.beta = 0 # Brightness control (0-100)
    self.position = (10,50)
    self.kernel_morphgrad = np.ones((2,2),np.uint8)
    self.thresholdCannyMin = 18#30 52 150 8 13 17
    self.thresholdCannyMax = 2
    self.count = 0
    self.while_needcontinue = False
    self.startPos_frames = 0
    self.startPos_milis = 0
    #endPos = 60000
    self.endPos_frames = 500
    self.endPos_milis = 60000/4
  
  def setTestPath(self,testpath):
    self.testpath = testpath
  
  ###algo1 funcs:
  def Contraste(self,tuple_img_alpha_beta):
    print("Contraste:")
    img, alpha, beta = tuple_img_alpha_beta
    self.im_contrast = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

  def Gray(self,img):
    print("Gray")
    self.im_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

  def Xor(self,img):
    print("Xor")
    self.im_xor = cv.bitwise_not(img)

  def Canny(self,img):
    print("Canny")
    self.im_canny = cv.Canny( img, self.thresholdCannyMin, self.thresholdCannyMax, 3 )

  def Dilate(self,img,shape):
    print("Dilate")
    im_height, im_width, im_channels = shape
    kernel_dilate = np.ones((im_height,im_width,3), np.uint8)
    kernel_dilate_blank = kernel_dilate[0:im_height,0:im_width] = (255,255,255)
    self.im_dilate = cv.dilate(img, kernel_dilate_blank, iterations = 3 )

  def MorphGradient1(self,img):
    print("MorphGradient1")
    self.im_morphgrad1 = cv.morphologyEx(img, cv.MORPH_GRADIENT, self.kernel_morphgrad)

  def MorphGradient2(self,img):
    print("MorphGradient2")
    self.im_morphgrad2 = cv.morphologyEx(img, cv.MORPH_GRADIENT, self.kernel_morphgrad)

  def MorphClose(self,img):
    print("MorphClose")
    self.im_morphclose = cv.morphologyEx(img, cv.MORPH_CLOSE, self.kernel_morphgrad)

  def Contours(self,img,img_original):
    contours_internal, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    self.imgOriginal_contours = img_original
    if len(contours_internal) > 0:
      for contour in contours_internal:
        #print(cv.contourArea(contour))
        if cv.contourArea(contour) <= 0.0:
          continue
        self.imgOriginal_contours = cv.drawContours(self.imgOriginal_contours, contour, 3, (0,255,0), 3)


  def algo1(self,tuple_img_nframe):
    img, nFrame = tuple_img_nframe
    img_name = ""
    self.Contraste( (img,self.alpha,self.beta) )
    img_path = self.testpath+"/"+img_name+str(self.Contraste.__name__)+"_"+str(nFrame)+".jpg"
    im_height, im_width, im_channels = img.shape
    imgToWrite = self.im_contrast
    if not cv.imwrite(img_path, imgToWrite):
     raise Exception("No se pudo guardar imagen")
    #----------------------------------------------------------
    self.Gray(self.im_contrast)
    img_path = self.testpath+"/"+img_name+str(self.Gray.__name__)+"_"+str(nFrame)+".jpg"
    imgToWrite = self.im_gray
    if not cv.imwrite(img_path, imgToWrite):
     raise Exception("No se pudo guardar imagen")
    #----------------------------------------------------------
    self.Xor(self.im_gray)
    img_path = self.testpath+"/"+img_name+str(self.Xor.__name__)+"_"+str(nFrame)+".jpg"
    imgToWrite = self.im_xor
    if not cv.imwrite(img_path, imgToWrite):
     raise Exception("No se pudo guardar imagen")
    #----------------------------------------------------------
    self.Canny(self.im_xor)
    img_path = self.testpath+"/"+img_name+str(self.Canny.__name__)+"_"+str(nFrame)+".jpg"
    imgToWrite = self.im_canny
    if not cv.imwrite(img_path, imgToWrite):
     raise Exception("No se pudo guardar imagen")
    #----------------------------------------------------------
    self.Dilate(  self.im_canny, img.shape )
    img_path = self.testpath+"/"+img_name+str(self.Dilate.__name__)+"_"+str(nFrame)+".jpg"
    imgToWrite = self.im_dilate
    if not cv.imwrite(img_path, imgToWrite):
     raise Exception("No se pudo guardar imagen")
    #----------------------------------------------------------
    self.MorphGradient1(self.im_dilate)
    img_path = self.testpath+"/"+img_name+str(self.MorphGradient1.__name__)+"_"+str(nFrame)+".jpg"
    imgToWrite = self.im_morphgrad1
    if not cv.imwrite(img_path, imgToWrite):
     raise Exception("No se pudo guardar imagen")
    #----------------------------------------------------------
    self.MorphClose(self.im_morphgrad1)
    img_path = self.testpath+"/"+img_name+str(self.MorphClose.__name__)+"_"+str(nFrame)+".jpg"
    imgToWrite = self.im_morphclose
    if not cv.imwrite(img_path, imgToWrite):
     raise Exception("No se pudo guardar imagen")
    #----------------------------------------------------------
    self.MorphGradient2(self.im_morphgrad1)
    img_path = self.testpath+"/"+img_name+str(self.MorphGradient2.__name__)+"_"+str(nFrame)+".jpg"
    imgToWrite = self.im_morphgrad2
    if not cv.imwrite(img_path, imgToWrite):
     raise Exception("No se pudo guardar imagen")
    #----------------------------------------------------------
    self.Contours(self.im_morphgrad1,img)
    img_path = self.testpath+"/"+img_name+str(self.Contours.__name__)+"_"+str(nFrame)+".jpg"
    imgToWrite = self.imgOriginal_contours
    if not cv.imwrite(img_path, imgToWrite):
     raise Exception("No se pudo guardar imagen")
    #cv.putText(self.imgOriginal_contours,str(self.count),self.position,cv.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),3)#numpy array on which text is written,text,position at which writing has to start,font family,font size,font color,font stroke  
    return

  #pasoporpaso = True
  def printMenu():
    print('''Menu:
    N.Siguiente frame
    F.Buscar un frame en especifico
    0.Siguiente estado''')
  def showDebug(title,im):
      cv.imshow(title, im)
      printMenu()
#----------------------------------------------------------
def readVideo():
  todoslosframes = []
  countReadVideo = 0
  startPos_frames = 0
  startPos_milis = 0
  endPos_frames = 500
  endPos_milis = 60000/4
  #endPos_milis = 60000/12
  #endPos_milis = 60000/24
  #endPos_milis = 60000/48
  print("countReadVideo: "+str(countReadVideo))
  print("startPos_milis: "+str(startPos_milis))
  print("endPos_milis: "+str(endPos_milis))
  video1path = r'video1.mp4'
  vidcap = cv.VideoCapture( video1path )
  #otro metodo con tresh!:
  vidcap.set(cv.CAP_PROP_POS_MSEC,startPos_milis)
  success,im = vidcap.read()
  nMillis = vidcap.get(cv.CAP_PROP_POS_MSEC)
  print("nMillis startPos "+str(nMillis))
  while success:
    cv.destroyAllWindows()
    print("countReadVideo: "+str(countReadVideo))
    nMillis = vidcap.get(cv.CAP_PROP_POS_MSEC)
    print("nMillis "+str(nMillis))
    vidcap.set(cv.CAP_PROP_POS_FRAMES,countReadVideo)
    success,im = vidcap.read()
    #self.todoslosframes.append(self.img_with_contours)
    todoslosframes.append(im)
    #img_final = self.img_with_contours
    #framepath = self.testpath + r'\frame'+str(countReadVideo)+'.jpg'
    #cv.imwrite(framepath, img_final)     # save frame as JPEG file
    countReadVideo += 1
    if endPos_milis != None:
      if nMillis >= endPos_milis:
        break
  print("todoslosframes len "+str( len(todoslosframes) ))
  return todoslosframes
#----------------------------------------------------------     
def showTodosLosFrames(todoslosframes):
  for frame in todoslosframes:
    cv.imshow('Frames', frame)
    cv.waitKey(1)
    time.sleep(0.03)#24frames

def run():
  root = []
  dirs = []
  files = []
  testCount = 0
  testRE = re.compile("^test([0-9]+)+$")
  decimalRE = re.compile(r'\d+')
  thispath_gen = os.walk('.')
  root,dirs,files = next(thispath_gen)
  print("root: "+str(root))
  print("dirs: "+str(dirs))
  print("files: "+str(files))
  testPaths = list(filter(testRE.match, dirs))
  testPaths = list(filter(os.path.isdir, testPaths))
  nTests = re.findall(decimalRE, str(testPaths))
  print("nTests: "+str(nTests))
  if len(nTests) > 0:
    testCount = max(nTests)
    testCount = int(testCount)+1
  testpath = r'test'+str(testCount)
  print("testpath: "+testpath)
  os.mkdir(testpath)
  testpath = os.path.abspath(testpath)#obtiene path absoluto
  print("Leyendo Video...")
  todoslosframes = readVideo()
  nFramesProcesados = 0
  PROCESSES = multiprocessing.cpu_count()# - 1
  print(f"Running with {PROCESSES} processes!")  
  start = time.time()
  with multiprocessing.Pool(PROCESSES) as p:
    scanner = Scanner()
    scanner.setTestPath(testpath)
    list_tuples_img_nframe = []
    for frame,nFrame in zip( todoslosframes,range( len(todoslosframes)-1 ) ):
      list_tuples_img_nframe.append( (frame,nFrame) )
    task1_result = p.map_async( scanner.algo1
                        ,list_tuples_img_nframe )
    #img_test = task1_result.get()
    p.close()
    p.join()
  print(f"Time taken = {time.time() - start:.10f}")


if __name__ == "__main__":
  run()
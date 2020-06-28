from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
from char_segmentation.extrPers import RunPersistence
from char_segmentation.path import a_star

## Check downwards from x for free path
def checkFreeYPath(im, x, height):
  for y in range(height):
    if (im[y][x] == 0):
      return False

  return True

## check sideways from Y for free path
def checkFreeXPath(im, y, width):
  for x in range(width):
    if (im[y][x] == 0):
      # print("not free x path found")
      return False

  return True


## Make initial cut, based on peristence IF path is free
def cutSegments(im, extrema_persistence, h):
  segments = []

  for x in extrema_persistence:
    if(checkFreeYPath(im,x,h)):
      s1 = im[:, :x].copy()
      s2 = im[:, x:].copy()
      im = s1
      s2 = cutWhite(s2, h)
      segments.append(s2)

  return segments

## Start A* pathfinding
def makePath(im):
  npimg = np.array(im)
  h = im.shape[0]
  w = im.shape[1]
  path = a_star(npimg, (0, int(w/2)), (h-1, int(w/2)))

  return path

## Drawing function for path
def drawPath(im, path):
  for p in path:
    im[p[0]][p[1]] = 125

  return im


## Crop whitespace on x borders of segment
def cutWhite(im, h):
  l,r = findYBounds(im, h)
  return im[:, l:r]

## Find whitespace around segment
def findYBounds(im, h):
  lb = 0
  rb = im.shape[1]

  for x in range(im.shape[1]):
    if(not checkFreeYPath(im, x, h)):
      lb = x
      break

  for x2 in reversed(range(im.shape[1])):
    if(not checkFreeYPath(im, x2, h)):
      rb = x2
      break

  return lb, rb

## Crop segment based on path found with A*
def makeCut(orig, path):
  w = orig.shape[1]
  im1 = orig.copy()
  im2 = orig.copy()

  ## Fill remaining pixels with WHITE for both segments based on path coordinates
  for c in path:
    for x in range(w):
      if x > c[1]:
        im1[c[0]][x] = 255
      elif x < c[1]:
        im2[c[0]][x] = 255

  return im1, im2

## Find extrema and respective persistance of whole text line
def pers(im, width):

  profile = np.zeros((width))

  for w in range(width):
    profile[w] = (im[:,w] == 0).sum()

  removed_zeros = profile[~np.all(profile == 0)]
  THRESHOLD = abs(np.mean(removed_zeros))

  extrema_persistence = RunPersistence(profile)
  extrema_persistence = [t[0] for t in extrema_persistence if t[1] > THRESHOLD]
  extrema_persistence.sort(reverse = True)

  return extrema_persistence

## Find extrema in bigram segment
def bigramPers(im, width, height):

  profile = np.zeros((width))

  for w in range(width):
    profile[w] = (im[:,w] == 0).sum()

  extrema_persistence = RunPersistence(profile)
  extrema_persistence = [t[0] for t in extrema_persistence if t[1] > 30]
  extrema_persistence.sort(reverse = True)

  return extrema_persistence

## Refine oversized segments using pathfinding
def refineSegm(segm, h):
  new = []
  ## Keep track of newly added segments to re-address these later on
  newSegm = 0

  for s in segm:
    width = s.shape[1]
    if width > 60:

      path = makePath(s)

      ## path returns 0 when no path has been found
      if(path != 0):
        im1, im2 = makeCut(s,path)
        im1 = cutWhite(im1, h)
        im2 = cutWhite(im2, h)

        ## Make sure segments are larger then .25* width to avoid small segments
        if im1.shape[1] > 0.25*s.shape[1] and im2.shape[1] > 0.25*s.shape[1]:
          new.append(im2)
          new.append(im1)
          newSegm += 1
        else:
          new.append(s)

      ## segm too wide but path not found -> bigram
      elif(width > 85):
        start = int((width/2)-20)
        end = int((width/2)+20)
        extr = bigramPers(s[:,start:end], end-start, h)

        if extr == []:
          new.append(s)
        elif len(extr) == 1:
          extr = extr[0]
          new.append(s[:,extr+start:])
          new.append(s[:,:extr+start])
          newSegm +=1
        elif len(extr) > 1:
          ## Take extrema closest to middle of segment
          extr = int(min(extr, key=lambda x:abs(x-width)))
          new.append(s[:,extr+start:])
          new.append(s[:,:extr+start])
          newSegm+=1
      else:
        new.append(s)
    else:
      new.append(s)

  return new, newSegm

## Remove empty segments and all-white segments
def cleanSegm(segm):
  temp = segm.copy()
  for s in segm:
    if s == []:
      temp.remove(s)
    elif np.mean(s) == 255:
      temp.remove(s)

  return temp

## Remove white space above and below segment
def cropY(segm):
  h = segm.shape[0]
  w = segm.shape[1]
  upB = 0
  lowB = h

  for y in range(h):
    if not checkFreeXPath(segm,y,w):
      upB = y
      break
  for y2 in reversed(range(h)):
    if not checkFreeXPath(segm, y2, w):
      lowB = y2
      break
    
  return segm[upB:lowB, :]

## Add padding to make segments resemble data set
def addBorders(s):
  size = 5
  new = cv2.copyMakeBorder(s.copy(),size,size,size,size,cv2.BORDER_CONSTANT,value=255)
  return new

## Main function for segmentation
def segmChars(line):
  h = line.shape[0]
  w = line.shape[1]

  ## Binarize image
  (thresh, line) = cv2.threshold(line, 127, 255, cv2.THRESH_BINARY)

  ext_pers = pers(line,w)

  ## Initial rough cut
  segm = cutSegments(line,ext_pers,h)

  segm = cleanSegm(segm)

  newSegm = -1
  maxSegmenting = 0

  ## Refining
  while newSegm != 0 and maxSegmenting != 30:
    segm, newSegm = refineSegm(segm, h)
    maxSegmenting += 1 
  
  ## Remove whitespace and add padding
  temp = []
  for s in segm:
    sTemp = cropY(s)
    sTemp = addBorders(sTemp)
    temp.append(sTemp)

  segm = temp

  return segm
from extrPers import RunPersistence
from path import a_star
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

im = cv2.imread("line.jpg", 0)
height = im.shape[0]  #  = 267
width = im.shape[1]   #  = 2583

## Binarize image
(thresh, im) = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

## Check downwards from x for free path
def checkFreePath(im, x):

  for y in range(height):
    if (im[y][x] == 0):
      return False

  return True

## Make initial cut, based on peristence IF path is free
def cutSegments(im, extrema_persistence):
  segments = []
  stopped = []

  for x in extrema_persistence:
    if(checkFreePath(im,x)):
      s1 = im[:, :x].copy()
      s2 = im[:, x:].copy()
      im = s1
      s2 = cutWhite(s2)
      segments.append(s2)
    else:
      stopped.append(x)

  return segments, stopped

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

## Crop whitespace on borders of segment
def cutWhite(im):
  l,r = findBounds(im)
  return im[:, l:r]

## Find whitespace around segment
def findBounds(im):
  lb = 0
  rb = im.shape[1]

  for x in range(im.shape[1]):
    if(not checkFreePath(im, x)):
      lb = x
      break

  for x2 in reversed(range(im.shape[1])):
    if(not checkFreePath(im, x2)):
      rb = x2
      break
  return lb, rb

## Crop segment based on path found with A*
def makeCut(orig, path):
  w = orig.shape[1]
  im1 = orig.copy()
  im2 = orig.copy()

  ## Fill remaining pixels with WHITE for both segments
  for c in path:
    for x in range(w):
      if x > c[1]:
        im1[c[0]][x] = 255
      elif x < c[1]:
        im2[c[0]][x] = 255

  return im1, im2

## Find extrema and respective persistance of whole text line
def pers(im):

  profile = np.zeros((width))

  for w in range(width):
    profile[w] = (im[:,w] == 0).sum()

  removed_zeros = profile[~np.all(profile == 0)]
  THRESHOLD = abs(np.mean(removed_zeros))

  # print(f'THRESHOLD: {THRESHOLD}')

  extrema_persistence = RunPersistence(profile)
  extrema_persistence = [t[0] for t in extrema_persistence if t[1] > THRESHOLD]
  extrema_persistence.sort(reverse = True)
  # print(len(extrema_persistence))

  # for idx in range(len(extrema_persistence)):
  #   if idx % 2 == 0:  
  #     x = extrema_persistence[idx]
  #     for y in range(height):
  #       im[y][x] = 125
  # cv2.imshow("im",im)
  # cv2.waitKey(0)

  return extrema_persistence

## Refine segments deemed to large using pathfinding
def refineSegm(segm):
  new = []

  for s in segm:
    if s.shape[1] > 90:

      path = makePath(s)

      # img = drawPath(s,path)

      im1, im2 = makeCut(s,path)
      im1 = cutWhite(im1)
      im2 = cutWhite(im2)
      new.append(im1)
      new.append(im2)

      # cv2.imshow("im", img)
      # cv2.waitKey(0)
    else:
      new.append(s)

  return new

ext_pers = pers(im)
segm, r = cutSegments(im,ext_pers)

# totW = 0
# totS = 0
# for s in segm:
#   totW += s.shape[1]
#   totS += 1
# avW = totW / totS

new = refineSegm(segm)
newer = refineSegm(new)

for s in newer:
    cv2.imshow("im",s)
    cv2.waitKey(0)  

# ret,thresh = cv2.threshold(ding,127,255,0)
# contours,hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# x,y,w,h = cv2.boundingRect(cnt)
# cv2.rectangle(ding,(x,y),(x+w,y+h),(0,255,0),2)
# rect = cv2.minAreaRect(cnt)
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(ding,[box],0,(0,0,255),2)
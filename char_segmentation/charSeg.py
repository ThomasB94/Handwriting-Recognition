from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
from char_segmentation.extrPers import RunPersistence
from char_segmentation.path import a_star
## Check downwards from x for free path
def checkFreePath(im, x, height):
  for y in range(height):
    if (im[y][x] == 0):
      return False

  return True

## Make initial cut, based on peristence IF path is free
def cutSegments(im, extrema_persistence, h):
  segments = []

  for x in extrema_persistence:
    if(checkFreePath(im,x,h)):
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

## Crop whitespace on borders of segment
def cutWhite(im, h):
  l,r = findBounds(im, h)

  return im[:, l:r]

## Find whitespace around segment
def findBounds(im, h):
  lb = 0
  rb = im.shape[1]

  for x in range(im.shape[1]):
    if(not checkFreePath(im, x, h)):
      lb = x
      break

  for x2 in reversed(range(im.shape[1])):
    if(not checkFreePath(im, x2, h)):
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

## Refine oversized segments using pathfinding
def refineSegm(segm, h):
  new = []
  newSegm = 0

  for s in segm:
    if s.shape[1] > 60:

      path = makePath(s)

      if(path != 0) :
        im1, im2 = makeCut(s,path)
        im1 = cutWhite(im1, h)
        im2 = cutWhite(im2, h)
        if im1.shape[1] > 0.25*s.shape[1] and im2.shape[1] > 0.25*s.shape[1]:
          new.append(im2)
          new.append(im1)
          # cv2.imshow("im", s)
          # cv2.waitKey(0)
          newSegm += 1
        else:
          new.append(s)
      else:
        new.append(s)
    else:
      new.append(s)

  return new, newSegm

def cleanSegm(segm):
  temp = segm.copy()
  for s in segm:
    if s == []:
      temp.remove(s)
    elif np.mean(s) == 255:
      temp.remove(s)

  return temp

def segmChars(line):
  h = line.shape[0]
  w = line.shape[1]

  ## Binarize image
  (thresh, line) = cv2.threshold(line, 127, 255, cv2.THRESH_BINARY)

  ext_pers = pers(line,w)

  segm = cutSegments(line,ext_pers,h)

  segm = cleanSegm(segm)

  newSegm = -1
  while newSegm != 0:
    segm, newSegm = refineSegm(segm, h)
  
  return segm
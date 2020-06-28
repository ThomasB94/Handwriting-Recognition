# -*- coding: utf-8 -*-

import numpy as np
import math
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os


def featureVect(image, features):
  #creates the feature vector of the images
  vector = [None] * len(features)
  for index, func in enumerate(features):
    vector[index] = np.array(func[0](image), dtype=np.float32)
  #flatten vector
  vector = np.reshape(vector, (-1))
  vector = np.concatenate(vector, axis=0)
  return vector

"""# Feature 1
horizontal histogram 10 bins
"""

def createHist(image, bins):
  hist = [None] * bins
  height, width = image.shape
  width = width / bins
  for index in range(bins):
    #start and end of the slices, floored so we get an index
    start = np.int(np.floor(width*index))
    end = np.int(np.floor(width*(index+1)))
    #slice the image
    roi = image[0:height, start:end]
    #to count the black pixels, we want the pixels that are zero
    hist[index] = ((end - start) * height) - cv2.countNonZero(roi)
  hist = [value / image.shape[1]*height for value in hist]
  return hist

def feature1(image):
  return createHist(image, CART_BINS)


"""# Feature 2
Verticle histogram 10 bins
"""

#uses definition of feature 1
def feature2(image):
  #transpose the image and run it through the hist
  transpose = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
  return createHist(transpose, CART_BINS)

"""# Feature 3
helper functions
"""

def polarConversion(image, centreX, centreY):
  #calculates polar conversion of image
  inverted = cv2.bitwise_not(image)
  radius = np.int(np.sqrt((image.shape[0] ** 2) + (image.shape[1] ** 2)) / 2)
  polar_image_inv = cv2.linearPolar(inverted,(centreX, centreY), radius, cv2.WARP_FILL_OUTLIERS)
  return cv2.bitwise_not(polar_image_inv)

def CalcCenter(image):
  #calculates centre of images
  inverted = cv2.bitwise_not(image)
  moment = cv2.moments(inverted)
  #Cannot find centre is zero. Approximate to centre of image
  if moment["m00"] == 0:
    return (image.shape[1] / 2, image.shape[0] / 2)
  cX = int(moment["m10"] / moment["m00"])
  cY = int(moment["m01"] / moment["m00"])
  return (cX, cY)

"""### Feature 3 radial slices histogram
a histogram of pie slices
"""

def feature3(image):
  #find centre
  (cx, cy) = CalcCenter(image)
  #polar representation around centre
  polar = polarConversion(image, cx, cy)

  #transpose the image and run it through the hist
  transpose = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
  return createHist(transpose, POLAR_BINS)

"""# Feature 4
radial bands histogram
a histogram of rings
"""

def feature4(image):
  #find centre
  (cx, cy) = CalcCenter(image)
  #polar representation around centre
  polar = polarConversion(image, cx, cy)

  #make a histogram of the polar version
  return createHist(polar, POLAR_BINS)

"""# Feature 5
radial out in profile
"""

def feature5(image):
  (cx, cy) = CalcCenter(image)
  #polar representation around centre
  polar = polarConversion(image, cx, cy)
  values = [None] * POLAR_ANGLE_NUM
  #Allong all the polar lines, go from out till insided, until you reach a value
  for index, location in zip(range(POLAR_ANGLE_NUM), np.linspace(0, polar.shape[0]-1, POLAR_ANGLE_NUM)):
    pos = polar.shape[1] - 1
    while pos > 0 and image[np.int(np.floor(location)), pos] == 255:
      pos = pos - 1
    values[index] = polar.shape[1] - pos
  values = [lines / polar.shape[1] for lines in values]
  return values

"""# Feature 6
Crossings
"""

def calcCrossingsOnLine(image, start, end):
  #calculates the amount of crossings in the image on a certain line
  points = np.int(np.floor(np.linalg.norm(start - end)))
  crossings = 0
  value = None
  for p in np.linspace(start, end, points):
    if image[tuple(np.int32(p))] != value:
      crossings = crossings + 1
      value = image[tuple(np.int32(p))]
  return crossings

def feature6(image):
  #calculates the crossings for diagonal lines radiating out from the centre of the image
  (cx, cy) = CalcCenter(image)
  crossingList = [None] * 4

  #vertical crossings
  start = np.array([0, cx])
  end = np.array([image.shape[0]-1, cx])
  crossingList[0] = calcCrossingsOnLine(image, start, end)

  #horizontal line
  start = np.array([cy, 0])
  end = np.array([cy, image.shape[1]-1])
  crossingList[1] = calcCrossingsOnLine(image, start, end)

  #diagonal line
  if cx >= cy:
    start = np.array([0, (cx - cy)])
  else:
    start = np.array([(cy - cx), 0])
  if (image.shape[1] - cx) >= (image.shape[0] - cy):
    end = np.array([image.shape[0] - 1, cx + image.shape[0] - cy - 1])
  else:
    end = np.array([cy + image.shape[1] - cx - 1, image.shape[1] - 1])
  crossingList[2] = calcCrossingsOnLine(image, start, end)

  #second diagonal line
  if cx + cy >= image.shape[1]:
    end = np.array([(cx + cy - image.shape[1] + 1), image.shape[1] - 1])
  else:
    end = np.array([0, (cy + cx)])

  if cx + cy >= image.shape[0]:
    start = np.array([image.shape[0] - 1, (cx + cy - image.shape[0] + 1)])
  else:
    start = np.array([(cy + cx), 0])
  crossingList[3] = calcCrossingsOnLine(image, start, end)
  return crossingList


"""# Feature 7

Hu's moments (invariant to rotation)
"""

def feature7(image):
  #Calculates the hough moments
  central_moments = cv2.moments(image)
  hu_moments = cv2.HuMoments(central_moments)
  for i in range(0,7):
    # cannot take log of zero, zo make it a very small number
    if hu_moments[i] == 0:
      hu_moments[i] = 0.000000001
    hu_moments[i] = -1* math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))
  hu_moments = np.reshape(hu_moments, (-1))
  return hu_moments

"""# Feature 8

Best houghlines
"""

def feature8(image):
  #inver to make the image foreground
  inverted = cv2.bitwise_not(image)
  #calculate houghline segments
  lines = cv2.HoughLinesP(inverted, 1, np.pi/180, 10)
  #maximum number of lines is 3
  if lines is None:
    return [0, 0, 0, 0]
  #if no lines were detected, return zeroes array.
  lines = lines[:1]
  #reshape array into 2d array
  lines = np.reshape(lines, (-1))
  #rescale values based on image sizes
  lines[0] = lines[0] / image.shape[1]
  lines[1] = lines[1] / image.shape[0]
  lines[2] = lines[2] / image.shape[1]
  lines[3] = lines[3] / image.shape[0]
  return lines


def feature9(image):
  # Finds hinge features in the image
  hist = np.zeros((12,12))

  contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  for contour in contours:
    for index, point in enumerate(contour):
      #dereference point
      point = point[0]
      #take points forward and backwards 5 points along the contours
      forward = contour[(index + 5) % len(contour)][0]
      backwards = contour[(index - 5) % len(contour)][0]
      #calculate angles of hinges
      angle1 = np.arctan2(forward[0] - point[0], forward[1] - point[0])
      angle1 = np.rad2deg(angle1 % (2 * np.pi))
      angle2 = np.arctan2(backwards[0] - point[0], backwards[1] - point[0])
      angle2 = np.rad2deg(angle2 % (2 * np.pi))
      #adds hinge values to right place in histogram
      if angle2 > angle1:
        hist[np.int(np.floor(angle1/30))][np.int(np.floor(angle2/30))] = hist[np.int(np.floor(angle1/30))][np.int(np.floor(angle2/30))] + 1

  #turn histogram into list
  listForm = []
  for y in range(12):
    for x in range(y, 12):
      listForm.append(hist[y][x])
  return listForm

"""Cross feature variables"""

CART_BINS = 10
POLAR_BINS = 8
POLAR_ANGLE_NUM = 20

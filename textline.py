import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']= 90
import time
import cv2
import os
os.chdir("/Users/paulhofman/Documents/Studie/Handwriting Recognition/image-data")
from persistence import RunPersistence
from pathfinding import a_star

FILE_ID = 'P168-Fg016-R-C01-R01-binarized.jpg'
WHITE = 255
BLACK = 0

def remove_whitespace(image):
  rows, cols = np.where(image == 0)
  r1 = min(rows) - 2
  r2 = max(rows) + 2
  c1 = min(cols) - 2
  c2 = max(cols) + 2
  return image[r1:r2,c1:c2]

def crop_line(image, row):
  cut =  200
  if row - cut < 0:
    return image[0:row+cut, :], row 
  elif row + cut >= image.shape[0]:
    return image[row-cut:, :], cut
  else:
    return image[row-cut:row+cut, :], cut

im = cv2.imread(FILE_ID)
# To 2d format, because images are already binarized
working_im = im[:,:,1]
working_im = remove_whitespace(working_im)
width = working_im.shape[1]
height = working_im.shape[0]
profile = np.zeros((height,))
for h in range(height):
  profile[h] = (working_im[h] == 0).sum()
# x indices height
# y num. of black pixels
# plt.plot(np.linspace(0,height, height), profile)
# plt.show()
THRESHOLD = 100
extrema_persistence = RunPersistence(profile)
extrema_persistence = [t for t in extrema_persistence if t[1] > 120]
# Odd indexes are minima, even maxima
minima = []
for idx in range(len(extrema_persistence)):
  if idx % 2 == 0:  
    r = extrema_persistence[idx][0]
    minima.append(r)
    # for c in range(width):
    #   working_im[r][c] = BLACK

paths = []
for idx in range(3):
  m = minima[idx]
  path = a_star(working_im, (m,0), (m,width-1))
  paths.append(path)
for path in paths:
  for p in path:
    r = p[0]
    c = p[1]
    working_im[r][c] = BLACK

cv2.imshow('img',working_im)
cv2.waitKey(0) 
cv2.destroyAllWindows() 

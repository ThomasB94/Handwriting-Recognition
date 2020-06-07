import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']= 90
import time
import cv2
import os
os.chdir("/Users/paulhofman/Documents/Studie/Handwriting Recognition/image-data")
from persistence import RunPersistence
from pathfinding import a_star

FILE_ID = 'P564-Fg003-R-C01-R01-binarized.jpg'
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
    
    return image[:row+cut, :], 

  elif row + cut >= image.shape[0]:
    
    return image[row-200:, :]

  else:
    
    return image[row-200:row+200, :], 200

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
minima = []
for idx in range(len(extrema_persistence)):
  if idx % 2 == 0:  
    y = extrema_persistence[idx][0]
    minima.append(y)

#     for x in range(width):
#       working_im[y][x] = BLACK
# cv2.imshow('im', working_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
# # cv2.resizeWindow('img', (500, 500))
# # cv2.imshow('img',working_im)
# # cv2.waitKey(0) 
# # cv2.destroyAllWindows() 

minima.sort()
print(minima)
p1 = minima[4]
p2 = minima[5]
print(p1, p2)
working_im = working_im[0:1200, :]
path1 = a_star(working_im, (p1,0), (p1, working_im.shape[1]-1))
path2 = a_star(working_im, (p2,0), (p2, working_im.shape[1]-1))

for p in path1:
  working_im[p[0]][p[1]] = BLACK

for p in path2:
  working_im[p[0]][p[1]] = BLACK
  
cv2.imshow('img',working_im)
cv2.waitKey(0) 
cv2.destroyAllWindows() 

 
# working_im = crop_line(working_im, 516)
# start = (516, 10)
# goal = (516, working_im.shape[1] - 10)
# path = a_star(working_im, start, goal)
# for (r,c) in path:
#   working_im[r][c] = BLACK

# cv2.imshow('img',working_im)
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 

import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi']= 90
import time
from cv2 import cv2
import os
from statistics import mean, stdev
from line_segmentation.persistence import RunPersistence
from line_segmentation.pathfinding import a_star

WHITE = 255
BLACK = 0

def remove_whitespace(image):
  rows, cols = np.where(image == 0)
  r1 = min(rows)
  r2 = max(rows)
  c1 = min(cols)
  c2 = max(cols) 
  
  return image[r1:r2,c1:c2]

def crop_line(image, row):
  cut =  200
  if row - cut < 0:
    return image[0:row+cut, :], row 
  elif row + cut >= image.shape[0]:
    return image[row-cut:, :], cut
  else:
    return image[row-cut:row+cut, :], cut


def textlines(im):
  lines = []
  # Removing the surrounding whitespace to increase pathfinding speed
  im = remove_whitespace(im)
  width = im.shape[1]
  height = im.shape[0]

  print("Image with height:", height, "and width:", width)
  print("Creating horizontal projection profile")
  profile = np.zeros((height,))
  for h in range(height):
    profile[h] = (im[h] == 0).sum()

  print("Determining minima")
  extrema_persistence = RunPersistence(profile)
  no_zeros = profile[profile != 0]
  mn = np.mean(no_zeros)
  mx = max(no_zeros)
  std = stdev(no_zeros)
  # mn = mean(profile)
  # mx = max(profile)
  # std = stdev(profile)
  print(mx, mn, std)
  THRESHOLD = mn
  # TESTING WITH NO ZEROS MEAN 
  THRESHOLD = np.mean(no_zeros)
  print(THRESHOLD)
  extrema_persistence = [t[0] for t in extrema_persistence if t[1] > THRESHOLD]
  # Odd indexes are minima, even maxima, we use only minima, so remove maxima
  minima = []
  for idx in range(len(extrema_persistence)):
    if idx % 2 == 0:  
      r = extrema_persistence[idx]
      minima.append(r)

  minima.sort()
  
  # check to see if there is a line at the top or at the bottom, we don't use those, so we remove them
  first_line = minima[0]
  last_line = minima[-1]
  if first_line < 20:
    minima.remove(first_line)
  if height - last_line < 20:
    minima.remove(last_line)

  ##########################################################################
  # this is just for drawing the found lines 
  # show_im = im.copy()
  # for m in minima:
  #   show_im[m-5:m+5,:] = BLACK
  # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
  # cv2.resizeWindow('img', 900,900)
  # cv2.imshow('img',show_im)
  # cv2.waitKey(0) 
  # cv2.destroyAllWindows() 

  
  ##########################################################################

  print("Found", len(minima), "minima with threshold =", THRESHOLD)
  print("Minima are:", minima)
  print("Determining path for every minimum")
  paths = []
  for m in minima:
    path = a_star(im, (m,0), (m,width-1))
    paths.append(path)

  ##########################################################################
  # this is just for drawing the found lines 
  # path_im = im.copy()
  # for path in paths:
  #   for p in path:
  #     r = p[0]
  #     c = p[1]
  #     path_im[r-3:r+3,c] = BLACK

  # cv2.imshow('img',path_im)
  # cv2.waitKey(0) 
  # cv2.destroyAllWindows() 

  
  ##########################################################################


  print("Cutting textlines out of image")
  num_paths = len(paths)
  # if we have num_paths paths, we have num_paths+1 sentences
  for idx in range(num_paths+1):

    cropped = im.copy()
    
    # first line, so we only use one line
    if idx == 0:
      path = paths[idx]
      # determines size of rect
      max_r = max(path, key=lambda x: x[0])[0]      
      for p in path:
        r = p[0]
        c = p[1]
        cropped[r:,c] = WHITE
      cropped = cropped[0:max_r][0:]
    
    # this is the last line, so above the last text line
    elif idx == num_paths:
      path = paths[idx-1]
      min_r = min(path, key=lambda x: x[0])[0]
      for p in path:
        r = p[0]
        c = p[1]
        cropped[0:r,c] = WHITE
      cropped = cropped[min_r:][0:]
    
    # other regular lines that have to be cut by using two paths
    else:
      upper_path = paths[idx-1]
      bottom_path = paths[idx]
      min_r = min(upper_path, key=lambda x: x[0])[0]
      max_r = max(bottom_path, key=lambda x: x[0])[0]

      for p in upper_path:
        r = p[0]
        c = p[1]
        cropped[0:r,c] = WHITE
      for p in bottom_path:
        r = p[0]
        c = p[1]
        cropped[r:,c] = WHITE
      cropped = cropped[min_r:max_r][0:]

    print("MEAN", np.mean(cropped), "HEIGHT:", cropped.shape)
    # cv2.imshow('img',cropped)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows() 
    # after cutting out a sentence, we check to see if it is actually a sentence
    # if it just a white rectangle we don't pass it to char seg
    # we also check to see if it can't possibly be a good sentence, because it is too small
    if not np.mean(cropped) > 250 and not cropped.shape[0] < 30:
      lines.append(cropped)      
      print("ADDED")
  
  print("Created all rectangles with sentences, now exiting line segmentation")
  return lines

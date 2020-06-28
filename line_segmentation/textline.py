import numpy as np
import time
from cv2 import cv2
import os
from statistics import mean, stdev
from line_segmentation.persistence import RunPersistence
from line_segmentation.pathfinding import a_star

WHITE = 255
BLACK = 0

# crops the image by removing all whitespace
def remove_whitespace(image):
  rows, cols = np.where(image == 0)
  r1 = min(rows)
  r2 = max(rows)
  c1 = min(cols)
  c2 = max(cols) 
  
  return image[r1:r2,c1:c2]

def textlines(im):
  lines = []
  # remove the surrounding whitespace to increase path finding speed
  im = remove_whitespace(im)
  width = im.shape[1]
  height = im.shape[0]

  print("Image with height:", height, "and width:", width)
  print("Creating horizontal projection profile")
  # creates horizontal projection profile by summing all black ink in a row of pixels
  profile = np.zeros((height,))
  for h in range(height):
    profile[h] = (im[h] == 0).sum()

  # use the peristence class retrieved from https://www.csc.kth.se/~weinkauf/notes/persistence1d.html to determine
  # persistence, which is the distance in height between extreme values of the projection profile
  print("Determining minima")
  extrema_persistence = RunPersistence(profile)

  # remove all horizontal projections which are 0
  no_zeros = profile[profile != 0]
  # the mean turns out to be a good threshold for persistence
  mn = np.mean(no_zeros)
  THRESHOLD = np.mean(no_zeros)
  extrema_persistence = [t[0] for t in extrema_persistence if t[1] > THRESHOLD]
  # odd indexes are minima, even maxima, we use only minima, so remove maxima
  minima = []
  for idx in range(len(extrema_persistence)):
    if idx % 2 == 0:  
      r = extrema_persistence[idx]
      minima.append(r)

  # sort the minima to make sure it goes from top of the page to bottom
  minima.sort()
  
  # check to see if there is a line at the top or at the bottom, we don't use those, so we remove them
  first_line = minima[0]
  last_line = minima[-1]
  if first_line < 20:
    minima.remove(first_line)
  if height - last_line < 20:
    minima.remove(last_line)

  print("Found", len(minima), "minima with threshold =", THRESHOLD)
  print("Minima are:", minima)
  print("Determining path for every minimum")
  paths = []
  # we creates path using the minima retrieved from persistence class
  # a star is used to navigate through the whitespace between sentences
  for m in minima:
    path = a_star(im, (m,0), (m,width-1))
    paths.append(path)

  print("Cutting textlines out of image")
  num_paths = len(paths)
  # the paths have been determined, now the rectangles with sentences can be cut out
  # if we have num_paths paths, we have num_paths+1 sentences
  for idx in range(num_paths+1):

    cropped = im.copy()
    
    # first line, so we only use one line to create the first rectangle
    if idx == 0:
      path = paths[idx]
      # determines size of rect
      max_r = max(path, key=lambda x: x[0])[0]      
      for p in path:
        r = p[0]
        c = p[1]
        cropped[r:,c] = WHITE
      cropped = cropped[0:max_r][0:]
    
    # this is the last line, so above the last text line, so again we use only one line
    elif idx == num_paths:
      path = paths[idx-1]
      min_r = min(path, key=lambda x: x[0])[0]
      for p in path:
        r = p[0]
        c = p[1]
        cropped[0:r,c] = WHITE
      cropped = cropped[min_r:][0:]
    
    # other regular lines that have to be cut by using two paths to create a rectangle
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

    # after cutting out a sentence, we check to see if it is actually a sentence
    # if it just a white rectangle we don't pass it to char seg
    # we also check to see if it can't possibly be a good sentence, because it is too small
    if not np.mean(cropped) > 250 and not cropped.shape[0] < 30:
      lines.append(cropped)      
  
  print("Created all rectangles with sentences, now exiting line segmentation")
  return lines

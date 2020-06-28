from line_segmentation.textline import textlines


import os
import cv2
os.chdir("/Users/paulhofman/Documents/Studie/Handwriting Recognition/image-data")

im = cv2.imread("P168-Fg016-R-C01-R01-binarized.jpg", cv2.IMREAD_GRAYSCALE)
lines = textlines(im)
counter = 1
for line in lines:
    print(counter)
    cv2.imshow('img',line)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    counter = counter + 1
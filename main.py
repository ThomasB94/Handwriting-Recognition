# regular imports
import os
import cv2

# our algorithms
from line_segmentation.textline import textlines

def main():
    # os stuff to walk through dir.

    for file_name in files:
        im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        # INPUT: cv2 grayscale image
        # OUTPUT: list of rectangular cv2 grayscale images that represent sentences
        lines = textlines(im)

        for line in lines:
            charList = segmChars(line)


        #lines = textlines(im)
        # for line in lines:
            #charList = segmChars(line)
        #class.
        #style
    
main()

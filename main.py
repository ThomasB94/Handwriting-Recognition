# regular imports
import os
from cv2 import cv2
import sys

# Line segmentation
from line_segmentation.textline import textlines

# Character segmentation
from char_segmentation.charSeg import segmChars

# Character recognition
from recognition.recognizers import Recognizer
from recognition import *
# Style classification
def main():
    path = str(sys.argv[1])
    print("Using the images found in the folder with path:", path)
    # get all the files in the directory, this should be the directory with the test images    
    files = os.listdir(path)
    # filter out everything that's not a JPEG image
    #################################################
    #TODO: MAKE SURE THIS ACTUALLY TAKES CORRECT EXT.
    #TODO: ALSO MAKE SURE THERE ARE NO SPACES IN THE PATH OR USE \
    ##################################################
    files = [f for f in files if f.lower().endswith('.jpg')]
    print("Found the following files:", files)
    for file_name in files:
        print(os.path.join(path,file_name))
        im = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_GRAYSCALE)
        print(im)
        # INPUT: cv2 grayscale image
        # OUTPUT: list of rectangular cv2 grayscale images that represent sentences
        lines = textlines(im)

        for line in lines:
            charList = segmChars(line)
            for ch in charList:
                cv2.imshow('image', ch)
                cv2.waitKey(0)
                cv2.destroyAllWindows() 
            print('done with 1 line')
        print('done with all lines')

        #lines = textlines(im)
        # for line in lines:
            #charList = segmChars(line)
        #class.
        #style

if __name__ == "__main__":
    print('main')
    recognizer = Recognizer()
    main()
    


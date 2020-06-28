# regular imports
import os
import cv2
import sys

# our algorithms
from line_segmentation.textline import textlines


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

if __name__ == "__main__":
    print('main')
    recognizer = Recognizer()
    main()
    


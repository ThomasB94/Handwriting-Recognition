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


alphabet_code = {"Alef":1,
                 "Bet":2,
                 "Gimel":3,
                 "Dalet":4,
                 "He":5,
                 "Waw":6,
                 "Zayin":7,
                 "Het":8,
                 "Tet":9,
                 "Yod":10,
                 "Kaf-final":11,
                 "Kaf":12,
                 "Lamed":13,
                 "Mem":14,
                 "Mem-medial":15,
                 "Nun-final":16,
                 "Nun-medial":17,
                 "Samekh":18,
                 "Ayin":19,
                 "Pe-final":20,
                 "Pe":21,
                 "Tsadi-final":22,
                 "Tsadi-medial":23,
                 "Qof":24,
                 "Resh":25,
                 "Shin":26,
                 "Taw":27}

hebrew_letters = [chr(letter) for letter in range(0x5d0, 0x5eb)]

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
    files = [f for f in files if f.lower().endswith('.pbm')]
    print("Found the following files:", files)
    for file_name in files:
        print(os.path.join(path,file_name))
        im = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_GRAYSCALE)
        # print(im)
        # INPUT: cv2 grayscale image
        # OUTPUT: list of rectangular cv2 grayscale images that represent sentences
        lines = textlines(im)

        sentences = []
        for line in lines:
            # print(line)
            charList = segmChars(line)
            recog_line = []
            for ch in charList:
                pred = recognizer.predict(ch)
                recog_line.append(pred)
                #cv2.imshow('img', ch)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

            hebrew_line = []
            recog_line.reverse()
            for c in recog_line:
                letter = hebrew_letters[alphabet_code[c]-1]
                # letter = letter.encode("utf-8")
                hebrew_line.append(letter)
            sentences.append(hebrew_line)

            print('done with 1 line')

        style = recognizer.get_style()
        result_file_name = file_name.split('.')[0]
        with open(os.path.join(path,result_file_name) + "_recog.txt", "w") as f:
            for sent in sentences:
                for c in sent:
                    f.write(c)
                f.write('\n')

        with open(os.path.join(path,result_file_name) + "_style.txt", "w") as f:
            f.write(style)

        print('done with all lines')
        break

if __name__ == "__main__":
    recognizer = Recognizer()
    main()

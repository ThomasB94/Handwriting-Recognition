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
    results_path = os.path.join(path, "results")
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    for file_name in files:
        try:
            print(os.path.join(path,file_name))
            im = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_GRAYSCALE)
            print("Line segmentation")
            print("------------------------")
            lines = textlines(im)
            sentences = []
            print("Character segmentation")
            print("------------------------")
            for line in lines:
                # print(line)
                try:
                    charList = segmChars(line)
                    recog_line = []
                    for ch in charList:
                        try:
                            pred = recognizer.predict(ch)
                            recog_line.append(pred)
                        except:
                            print("Skipping character because of error")

                    hebrew_line = []
                    for c in recog_line:
                        letter = hebrew_letters[alphabet_code[c]-1]
                        hebrew_line.append(letter)
                    sentences.append(hebrew_line)
                except:
                    print("Sentence segmentation went wrong, skipping line")
                    sentences.append([])

            print("Character classification")
            print("------------------------")

            print("Style classification")
            print("------------------------")
            style = recognizer.get_style()
            result_file_name = file_name.split('.')[0]
            with open(os.path.join(results_path,result_file_name) + "_characters.txt", "w") as f:
                for sent in sentences:
                    for c in sent:
                        f.write(c)
                    f.write('\n')

            f.close()

            with open(os.path.join(results_path,result_file_name) + "_style.txt", "w") as f:
                f.write(style)

            f.close()

            print('Finished with', file_name)
        except:
            print("Something went wrong with this page, going to the next one")

    print("Completely done, results are in the results folder with the images")


if __name__ == "__main__":
    recognizer = Recognizer()
    main()

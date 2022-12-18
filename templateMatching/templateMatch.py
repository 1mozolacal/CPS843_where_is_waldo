import cv2
import numpy as np

threshold =0.8

#matchModes = {"CCOEFF_NORM":cv2.TM_CCOEFF_NORMED}
#List of all normalized matching methods so that they can each individually be tested
matchModes = {"SQDIFF_NORM":cv2.TM_SQDIFF_NORMED,"CCORR_NORM":cv2.TM_CCORR_NORMED,"CCOEFF_NORM":cv2.TM_CCOEFF_NORMED}
templateImage = cv2.imread(r'.\wheres-waldo\manual\1.jpg')

for mode in matchModes:
    fullImage = cv2.imread(r'.\wheres-waldo\Hey-Waldo\original-images\1.jpg')
    #Run the template matching algorithm
    matching = cv2.matchTemplate(fullImage,templateImage,matchModes[mode])

    colors, width, height = templateImage.shape[::-1]
    matches = np.where( matching >= threshold)
    #Draw in green boxes any frame where the matching value was not less than the threshold value
    for match in zip(*matches[::-1]):
        cv2.rectangle(fullImage, match, (int(match[0] + width), int(match[1] + height)), (0,255,0), 2)
    #Write resulting image to a new file
    cv2.imwrite('.\\template_match_results\\'+mode+'.png',fullImage)

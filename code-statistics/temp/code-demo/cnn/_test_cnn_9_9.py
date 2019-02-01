import cv2 as cv

from __init__ import write_log, you_jpg

lg = "_test_cnn_9_9.log"

img = cv.imread(you_jpg)

hog = cv.HOGDescriptor()
foundLocations, foundWeights = hog.detectMultiScale(img)
foundLocations_d, weights = hog.detect(img)
descriptors = hog.compute(img)

write_log(str(foundLocations), file=lg)
write_log(str(foundLocations_d), file=lg)
write_log(str(descriptors), file=lg)

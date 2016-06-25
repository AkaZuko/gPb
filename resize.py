import cv2
import sys

# (width, height)
img = cv2.imread(sys.argv[1])
img = cv2.resize(img, (sys.argv[2], sys.argv[3]), interpolation = cv2.INTER_CUBIC)
cv2.imwrite(sys.argv[1], img)
import cv2
import sys

img = cv2.imread(sys.argv[1], 0)

colors = {}

height, width = img.shape
for row in range(height):
	for col in range(width):
		col = img[row][col]
		if col not in colors:
			colors[col] = 1
		else:
			colors[col] += 1
print colors
#cv2.imshow('test', img)
#cv2.waitKey(0)

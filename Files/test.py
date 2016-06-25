import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

im = cv2.imread(sys.argv[1])

height,width,depth = im.shape

for i in range(0, 360, 45):
	# print i, i+180
	circle_img_1 = np.zeros((height,width), np.uint8)
	circle_img_2 = np.zeros((height,width), np.uint8)
	
	cv2.ellipse(circle_img_1, (20, 25), (25,25), 0, (i), (i + 45), (255,255,255), thickness = -1)
	cv2.ellipse(circle_img_2, (120, 25), (25,25), 0, (i+180), (i + 180 + 180), (255,255,255), thickness = -1)
	
	masked_data_1 = cv2.bitwise_and(im, im, mask=circle_img_1)
	# masked_data_2 = cv2.bitwise_and(im, im, mask=circle_img_2)

	t1 = cv2.calcHist([masked_data_1], [0], circle_img_1, [256], [0,256])
	print t1.T
	# t2 = cv2.calcHist([masked_data_2], [0], circle_img_1, [256], [0,256])
	# print t1
	# print t2
	# gh = input()

	# lol = masked_data_1 + masked_data_2
	# cv2.imshow("masked_1", masked_data_1)
	# plt.hist(lol.ravel(),256,[0,256])
	# plt.show()
	# cv2.imshow("masked_2", circle_img_1)
	# cv2.waitKey(0)


# cv2.circle(circle_img,(width/2,height/2),100,1,thickness=-1)
# cv2.imshow("masked", im)
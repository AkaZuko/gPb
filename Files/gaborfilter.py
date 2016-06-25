#!/usr/bin/env python
 
import numpy as np
import cv2
 
def build_filters():
	filters = []
	ksize = 11
	for theta in np.arange(0, np.pi, np.pi / 16):
		kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
		kern /= 1.5*kern.sum()
		filters.append(kern)
	return filters
	# for f in filters:
	# 	f = cv2.resize(f, (100, 100), interpolation = cv2.INTER_CUBIC)
	# 	cv2.imshow('result', np.asarray(f))
	# 	cv2.waitKey(0)

def process(img, filters):
	accum = np.zeros_like(img)
	for kern in filters:
		fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
		cv2.imshow('result', np.asarray(fimg))
		cv2.waitKey(0)
		# np.maximum(accum, fimg, accum)
	# return accum

if __name__ == '__main__':
	import sys
	# build_filters()
	print __doc__
	try:
		img_fn = sys.argv[1]
	except:
		img_fn = 'test.png'
	img = cv2.imread(img_fn)
	img = cv2.resize(img, (100, 100), interpolation = cv2.INTER_CUBIC)
	if img is None:
		print 'Failed to load image file:', img_fn
		sys.exit(1)
	filters = build_filters()
	res1 = process(img, filters)
	# cv2.imshow('result', res1)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
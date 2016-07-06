import cv2
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

def edge_detection(img):
	print '*' * 50
	print 'edge_detection called'
	print '*' * 50
	print

	# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	blur = cv2.GaussianBlur(img,(3,3),0)
	edges = cv2.Canny(blur, 300, 400)

	kernel = np.ones((5,5), np.uint8)
	dilation = cv2.dilate(edges, kernel, iterations = 1)

	im_floodfill = dilation.copy()

	h, w = dilation.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)
	cv2.floodFill(im_floodfill, mask, (0,0), 255)

	im_floodfill_inv = cv2.bitwise_not(im_floodfill)

	im_out = dilation | im_floodfill_inv

	imgg = im_out.copy()
	

	contours, hierarchy = cv2.findContours(imgg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	boundary_points = []
	for i in range(len(contours)):
		cimg = np.zeros_like(img)
		cv2.drawContours(cimg, contours, i, color=255, thickness=1)
		pts = np.where(cimg == 255)
		bpoints = []
		for index in xrange(len(pts[0])):
			bpoints.append([pts[0][index], pts[1][index]])
		boundary_points.append(bpoints)
	
	# cv2.drawContours(imgg,contours,-1,color = 255,thickness = 1)
	# cv2.imshow('FGH', imgg)
	# cv2.waitKey(0)
	
	return boundary_points[0]
	# return np.reshape(contours[0].ravel(), (-1,2))

def make_invariant(points):
	print '*' * 50
	print 'make_invariant called'
	print '*' * 50
	print


	N = len(points)
	# points = map( lambda x : np.complex(x[0], x[1]), points)
	# FD = np.fft.fft(points)
	# FD1 = FD[1]
	# FD = FD[2:]
	# FD = map(lambda x : abs(x) / abs(FD1), FD)
	# FD = map(lambda x : round(x,1), FD)
	# return FD

	C  = map(lambda x  : np.asarray( [ [x[0]], [x[1]] ] ), points)
	C = np.asarray(C)
	print C.shape
	u = sum(C) / (N*1.0)
	sigma = sum(map(lambda x : (x-u)*((x-u).T) , C)) / (N*1.0)
	eigenvalues, eigenvectors  = np.linalg.eig(sigma)
	# inverseEigenVectors = np.linalg.inv(eigenvectors)
	eigenvectors_t = eigenvectors.T
	diagonal = eigenvectors_t.dot(sigma).dot(eigenvectors)
	# diagonal = diagonal.round(5)

	diagonal_i = diagonal.copy()
	for i in range(len(diagonal)):
		diagonal_i[i][i] = diagonal[i][i]**(-0.5)
	Cprime =  diagonal_i.dot(eigenvectors_t).dot(C - u)

	return Cprime.T[0]
	# return [0.0, 0.0]


temp_img = cv2.imread(sys.argv[1], 0)
height, width = temp_img.shape[:2]
points1 = make_invariant( edge_detection( temp_img ) )

temp_img2 = cv2.imread(sys.argv[2], 0)
height, width = temp_img2.shape[:2]
points2 = make_invariant( edge_detection( temp_img2 ) )

l1 = len(points1)
l2 = len(points2)
c = 0

plt.scatter(points1[:,0], points1[:,1])
plt.scatter(points2[:,0], points2[:,1], c = u'g')
plt.show()
# print points1
# print points2
for i in range(min(l1, l2)):
	if points1[i][0] == points2[i][0] and points1[i][1] == points2[i][1]:
		c += 1
		# print points1[i]	
		# same = False
		# break

print (c*100.0 / min(l1,l2))
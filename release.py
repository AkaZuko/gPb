import cv2
import sys
import math
import numpy as np

import matplotlib.pyplot as plt

class BezierCurve():
    def __init__(self, inp):
        self.points = np.asarray( map(lambda x : [x[0][0], x[1][0]], inp) )


class Bezier():
    """Find the piece-wise Bezier Curve"""
    def __init__(self, err):
        self.error =  err
        self.curves = []

    def get_control_points(self, points, start, end):
        N = end - start     
        P_O = np.asarray(points[start])
        P_3 = np.asarray(points[end])

        C_1 = np.asarray([[0.0],[0.0]])
        for i in range(N + 1):
            ti = i / (N*1.0)
            C_1 += 3 * ti * ((1-ti)**2) * ( points[start + i] - ((1-ti)**3)*P_O - (ti**3)*P_3 )

        C_2 = np.asarray([[0.0],[0.0]])
        for i in range(N + 1):
            ti = i / (N*1.0)
            C_2 += 3 * (1-ti) * ((ti)**2) * ( points[start + i] - ((1-ti)**3)*P_O - (ti**3)*P_3 )

        A_1 = 0.0
        for i in range(N + 1):
            ti = i / (N*1.0)
            A_1 += (ti**2) * ((1-ti)**4)
        A_1 *= 9

        A_2 = 0.0
        for i in range(N + 1):
            ti = i / (N*1.0)
            A_2 += (ti**4) * ((1-ti)**2)
        A_2 *= 9
        
        A_12 = 0.0
        for i in range(N + 1):
            ti = i / (N*1.0)
            A_12 += (ti**3) * ((1-ti)**3)
        A_12 *= 9
        
        P_1 = (A_2 * C_1 - A_12 * C_2) / (A_1 * A_2 - A_12 * A_12)
        P_2 = (A_1 * C_2 - A_12 * C_1) / (A_1 * A_2 - A_12 * A_12)

        return np.asarray([P_O, P_1, P_2, P_3])

    def get_point(self, CP, t):
        return ((1-t)**3)*CP[0] + 3*t*((1-t)**2)*CP[1] + 3*(1-t)*(t**2)*CP[2] + (t**3)*CP[3]  

    def distance(self, p1, p2):
        return ((p1[0][0] - p2[0][0])**2 + (p1[1][0] - p2[1][0])**2)**0.5

    def get_max_error_index(self, points, start, end, CP):
        error = 0
        index = start
        N = end - start + 1
        if N <= 4 :
            return -1

        for i in range(N):
            t = i / ((N-1)*1.0)
            point = self.get_point(CP, t)
            err = self.distance(points[start + i], point)
            if err > error : 
                error = err
                index = start + i
        if error < self.error:
            return -1 # just don't do any subdivision now
        return index

    def gen_curves(self, points, start, end):
        CP = self.get_control_points(points, start, end)
        index = self.get_max_error_index(points, start, end, CP)
        N = end - start + 1
        print 'N :', N
        if index == -1 or (index - start + 1) < 4 or (end - index + 1) < 4:
            self.curves.append(BezierCurve(CP))
            return

        self.gen_curves(points, start, index)
        self.gen_curves(points, index, end)
        return

    def plotBZ(self, CP):
        t = 0
        CPP = np.asarray(map(lambda x : [[x[0]], [x[1]]], CP))
        inc = 10**-2
        pts = []
        while True:
            point = self.get_point(CPP, t)
            pts.append([point[0][0], point[1][0]])
            t += inc
            if t >= 1:
                break
        return np.asarray(pts)


class Segmentation():

	def gen_mask(self, width, height, color_val, img_gray):
		print '*' * 50
		print 'gen_mask called'
		print '*' * 50
		print

		mask = img_gray.copy()

		for row in range(height):
			for col in range(width):
				if img_gray[row][col] == color_val:
					mask[row][col] = 255
				else:
					mask[row][col] = 0
		return mask

	def masked_image(self, img_gray, img_color, color_val):
		print '*' * 50
		print 'masked_image called'
		print '*' * 50
		print

		height, width = img_gray.shape
		mask_ = self.gen_mask(width, height, color_val, img_gray)
		masked_img = cv2.bitwise_and(img_color, img_color, mask = mask_)
		return np.asarray(masked_img)


	def gen_image(self, points, height, width):
		print '*' * 50
		print 'gen_image called'
		print '*' * 50
		print

		points_ = {}
		for i in points:
			state_ = str(i[0]) + " " + str(i[1])
			points_[state_] = 1

		img = np.zeros((height, width), np.uint8)
		for i in xrange(height):
			for j in xrange(width):
				state_ = str(i) + " " + str(j)
				if state_ in points_:
					img[i][j] = 255
		return img


	def edge_detection(self, img):
		print '*' * 50
		print 'edge_detection called'
		print '*' * 50
		print

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

		cv2.imshow('res', imgg)
		cv2.waitKey(0)

		contours, hierarchy = cv2.findContours(imgg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		max_area = -1*float('inf')
		boundary_points = [None]

		for i in range(len(contours)):
			area = cv2.contourArea(contours[i])

			if area > max_area:
				max_area = area
				cimg = np.zeros_like(img)
				cv2.drawContours(cimg, contours, i, color=255, thickness=1)
				pts = np.where(cimg == 255)
				bpoints = []
				for index in xrange(len(pts[0])):
					bpoints.append([pts[0][index], pts[1][index]])
				boundary_points[0] = np.asarray(bpoints)

		# print boundary_points[0].shape
		return boundary_points

	def make_invariant(self, points):
		print '*' * 50
		print 'make_invariant called'
		print '*' * 50
		print

		N = len(points)
		C  = map(lambda x  : np.asarray( [ [x[0]], [x[1]] ] ), points)
		C = np.asarray(C)
		u = sum(C) / (N*1.0)

		sigma = sum(map(lambda x : (x-u)*((x-u).T) , C)) / (N*1.0)
		eigenvalues, eigenvectors  = np.linalg.eig(sigma)
		eigenvectors_t = eigenvectors.T
		diagonal = eigenvectors_t.dot(sigma).dot(eigenvectors)
		# diagonal[diagonal<0.00001] = 0

		diagonal_i = diagonal.copy()
		for i in range(len(diagonal)):
			diagonal_i[i][i] = diagonal[i][i]**(-0.5)
		Cprime =  diagonal_i.dot(eigenvectors_t).dot(C - u)

		return Cprime.T[0]

	def process(self, path_seg, path_original, error_limit):
		print '*' * 50
		print 'process called'
		print '*' * 50
		print

		img_gray = cv2.imread(path_seg, 0)
		img_color = cv2.imread(path_original)
	
	
		height1, width1 = img_gray.shape
		height2, width2, channels = img_color.shape

		height = max(height1, height2)
		width = max(width1, width2)

		# img_gray = cv2.resize(img_gray, (width, height), interpolation = cv2.INTER_CUBIC)
		# img_color = cv2.resize(img_color, (width, height), interpolation = cv2.INTER_CUBIC)
	
		colors = np.unique(img_gray)

		for color in colors:
			masked_img = self.masked_image(img_gray, img_color, color)
			
			for boundary_points_ in self.edge_detection(masked_img):
				
				if boundary_points_ is None:
					continue
				
				boundary_points_ = np.asarray(map(lambda x : np.asarray([x[1], x[0]]), boundary_points_))

				boundary_points_inv = self.make_invariant(boundary_points_)
				points = np.asarray( map(lambda x : np.asarray([ [ x[0] ], [ x[1]] ]), boundary_points_inv) )
				bz = Bezier(error_limit)
				curves = bz.gen_curves(points, 0 , len(points) - 1)

				# points  = np.asarray(map(lambda x : [x[0][0], x[1][0]], points))
				plt.clf()
				# plt.figure(1)
				# plt.subplot(211)
				
				# plt.scatter(boundary_points_inv[:,0], boundary_points_inv[:,1])
				# plt.subplot(212)
				for curve in bz.curves:
					pts = bz.plotBZ(curve.points)
					plt.scatter(pts[:,0], pts[:,1], c = u'g')
					plt.scatter(curve.points[:,0], curve.points[:,1], c = u'g')
				plt.show()
				

ss = Segmentation()

# segmented_regions_image, original_image, error
ss.process(sys.argv[1], sys.argv[2], 10**-7)
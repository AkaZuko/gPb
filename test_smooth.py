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

        if index < start + 4:
            return start + 4
        elif index > end - 4:
            return end - 4

        return index

    def gen_curves(self, points, start, end):
        CP = self.get_control_points(points, start, end)

        N = end - start + 1
        if N<8:
            self.curves.append(BezierCurve(CP))
            return

        index = self.get_max_error_index(points, start, end, CP)
        # if index == -1 or (index - start + 1) < 4 or (end - index + 1) < 4:
        if index == -1:
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


	def dfs(self, cimg, x, y, h, w):
		
		s = []
		s.append([x,y])
		
		points = []
		points.append([x,y])
		
		vis = [ [False for W in range(w)] for H in range(h) ]
		vis[x][y] = True

		nexx = [[-1,-1], [-1,0], [-1,1], [0,1], [1,1], [1,0], [1,-1], [0,-1] ]		
		
		while s!=[]:
			x_,y_ = s.pop()
			for del_x, del_y in nexx:
				new_x , new_y = x_ + del_x, y_ + del_y
				if (new_x<h and new_x >=0) and (new_y<w and new_y>=0) and cimg[new_x][new_y]!=0 and vis[new_x][new_y]!=True:
					vis[new_x][new_y] = True
					s.append([new_x, new_y])
					points.append([new_x, new_y])

		return points

	def draw_image(self, boundary_points, width, height):
		img = np.zeros((height, width), np.uint8)
		for point in boundary_points:
			img[point[0]][point[1]] = 255
		cv2.imshow('Test Show', img)
		cv2.waitKey(0)

	def edge_detection(self, img, color):
		print '*' * 50
		print 'edge_detection called'
		print '*' * 50
		print

		imgg = img.copy()
		imgg[img==color] = 255
		imgg[img!=color] = 0

		h,w = img.shape[:2]
		# cv2.imshow('res', imgg)
		# cv2.waitKey(0)

		imgg = cv2.GaussianBlur(imgg,(17,17),0)
		# print 'blurring'
		# cv2.imshow('res', imgg)
		# cv2.waitKey(0)

		contours, hierarchy = cv2.findContours(imgg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		req_contour = None
		max_area = -1*float('inf')
		for i in range(len(contours)):
			area = cv2.contourArea(contours[i])
			if area > max_area:
				max_area = area
				req_contour = i
		
		cimg = np.zeros_like(img)
		cv2.drawContours(cimg, contours, req_contour, color=255, thickness=1)

		# cv2.imshow('test',cimg)
		# cv2.waitKey(0)

		for x in xrange(h):
			for y in xrange(w):
				if cimg[x][y] !=0:
					points = self.dfs(cimg, x, y, h, w) 
					points.append(points[0])
					print 'points', len(points)
					return [points]
		return [None]

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

		colors = np.unique(img_gray)
		print len(colors)
		for color in colors:
			for boundary_points_ in self.edge_detection(img_gray, color):

				if boundary_points_ is None:
					print 'No boundary points'
					continue

				points = np.asarray( map(lambda x : np.asarray([ [ x[0] ], [ x[1]] ]), boundary_points_ ) )
				
				bz = Bezier(error_limit)
				curves = bz.gen_curves(points, 0 , len(points) - 1)

				plt.clf()
				plt.xlim(xmax=250); plt.ylim(ymax=250);
				for curve in bz.curves:
					pts = bz.plotBZ(curve.points)
					plt.scatter(pts[:,0], pts[:,1], c = u'g')
					plt.scatter(curve.points[:,0], curve.points[:,1], c = u'y')
				plt.show()
				

ss = Segmentation()

# segmented_regions_image, original_image, error
ss.process(sys.argv[1], sys.argv[2], 1.999)
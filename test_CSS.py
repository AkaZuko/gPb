import cv2
import numpy as np
import sys
import pylab as plt

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

	def edge_detection(self, img, color, sigma):
		# print '*' * 50
		# print 'edge_detection called'
		# print '*' * 50
		# print

		imgg = img.copy()
		
		imgg[img==color] = 255
		imgg[img!=color] = 0

		h,w = img.shape[:2]
		# cv2.imshow('res', imgg)
		# cv2.waitKey(0)

		kernel = cv2.getGaussianKernel(11, sigma, ktype = cv2.CV_64F )
		imgg = cv2.filter2D(imgg, -1, kernel)
		# imgg = cv2.GaussianBlur(imgg,(17,17),0)
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
					# print 'points', len(points)
					return [points]
		return [None]

	def gen_css(self, image, color):
		plt.clf()
		
		sigma = 4.0	
		dsigma = 4.0

		while True:

			bdry = self.edge_detection(image, color, sigma)[0]

			if bdry is not None:
				bdry = np.asarray(bdry)

				dx_dt = np.gradient(bdry[:, 0])
				dy_dt = np.gradient(bdry[:, 1])
						
				d2x_dt2 = np.gradient(dx_dt)
				d2y_dt2 = np.gradient(dy_dt)

				curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5

				# option 1
				temp_ = np.sign(curvature)
				temp_[temp_==0] = -1
				zero_crossings = np.where(np.diff(temp_))[0]
				
				# option 2
				# zero_crossings = np.where(curvature==0)[0]

				if len(zero_crossings) == 0:
					break

				print 'len(zero_crossings)', len(zero_crossings)
				plot_data = np.asarray(map(lambda x : [ x, sigma ], zero_crossings ))
				plt.plot(plot_data[:,0], plot_data[:,1] ,marker='o', linestyle='None',label="test")
			
			else:
				break

			sigma += dsigma

		plt.show()

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

		#remove it later
		img_gray[img_gray!=0] = 255

		img_color = cv2.imread(path_original)
	
	
		height1, width1 = img_gray.shape
		height2, width2, channels = img_color.shape

		height = max(height1, height2)
		width = max(width1, width2)

		colors = np.unique(img_gray)
		print len(colors)

		for color in colors:
			# remove the if condition later
			if color != 255:
				self.gen_css(img_gray, color)			
			# for boundary_points_ in self.edge_detection(img_gray, color):

			# 	if boundary_points_ is None:
			# 		print 'No boundary points'
			# 		continue

			# 	points = np.asarray( map(lambda x : np.asarray([ [ x[0] ], [ x[1]] ]), boundary_points_ ) )
				
			# 	bz = Bezier(error_limit)
			# 	curves = bz.gen_curves(points, 0 , len(points) - 1)

			# 	plt.clf()
			# 	plt.xlim(xmax=250); plt.ylim(ymax=250);
			# 	for curve in bz.curves:
			# 		# pts = bz.plotBZ(curve.points)
			# 		# plt.scatter(pts[:,0], pts[:,1], c = u'g')
			# 		plt.scatter(curve.points[:,0], curve.points[:,1], c = u'y')
			# 	plt.show()
	

ss = Segmentation()

# segmented_regions_image, original_image, error
ss.process(sys.argv[1], sys.argv[2], 2.0)

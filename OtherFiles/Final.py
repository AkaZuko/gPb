import cv2
import sys
import math
import numpy as np

def gen_mask(width, height, color_val, img_gray):
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

def masked_image(img_gray, img_color, color_val):
	print '*' * 50
	print 'masked_image called'
	print '*' * 50
	print

	height, width = img_gray.shape
	mask_ = gen_mask(width, height, color_val, img_gray)
	masked_img = cv2.bitwise_and(img_color, img_color, mask = mask_)
	return np.asarray(masked_img)


def gen_image(points, height, width):
	print '*' * 50
	print 'gen_image called'
	print '*' * 50
	print
	
	# print points

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
				print 'TT'
	return img


def make_invariant(points):
	print '*' * 50
	print 'make_invariant called'
	print '*' * 50
	print

	N = len(points)

	points = map(lambda x : np.complex(x[0], x[1]), points)
	FD = np.fft.fft(points)
	# translational invariance
	translational_invar = FD[0]
	FD = map(lambda x : x - translational_invar, FD)
	# scalar invariance 
	val = ( ( FD[1] )*( FD[1].conjugate() ) ).real
	FD = map(lambda x : x / val, FD)
	# rotational invariance
	final_points = []
	for i in range(N):
		prev_ = FD[(i-1)%N]
		next_ = FD[(i+1)%N]
		angle = ( 2 * math.pi * i )/N
		cmplx1 = np.complex(math.cos(angle), math.sin(angle))
		cmplx2 = cmplx1.conjugate()
		u_i = prev_*cmplx2 + next_*cmplx1
		final_points.append(u_i)

	phi = math.atan( final_points[1].imag / final_points[1].real )
	cmplx3 = np.complex(math.cos(phi), math.sin(phi))

	final_points = map( lambda x : [ (x*cmplx3).real, (x*cmplx3).imag], final_points)	

	print final_points
	return final_points


def edge_detection(img):
	print '*' * 50
	print 'edge_detection called'
	print '*' * 50
	print

	img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
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

	contours, hierarchy = cv2.findContours(im_out,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	boundary_points = []
	for cnt in contours:
		# we can put some area threshold condition here
		boundary_points.append(np.reshape(cnt.ravel(), (-1, 2) ) )

	return boundary_points

def get_control_points(points_list):
	print '*' * 50
	print 'get_control_points called'
	print '*' * 50
	print
	
	# find the RHS Matrix value say V

	# to compute M^-1 use temp = np.linalg.inv(M)

	# result ie. [ [B_1], [B_2], ... , [B_n-1]]  = np.dot(temp, V)
	return


def process(path_seg, path_original):
	print '*' * 50
	print 'process called'
	print '*' * 50
	print

	img_gray = cv2.imread(path_seg, 0)
	img_color = cv2.imread(path_original)
	
	colors = {}
	
	height1, width1 = img_gray.shape
	height2, width2, channels = img_color.shape

	height = max(height1, height2)
	width = max(width1, width2)

	img_gray = cv2.resize(img_gray, (width, height), interpolation = cv2.INTER_CUBIC)
	img_color = cv2.resize(img_color, (width, height), interpolation = cv2.INTER_CUBIC)
	
	for row in range(height):
		for col in range(width):
			val = img_gray[row][col]
			if val in colors:
				colors[val] += 1
			else:
				colors[val] = 1

	for color in colors:
		masked_img = masked_image(img_gray, img_color, color)
		boundary_points = edge_detection(masked_img)		

process(sys.argv[1], sys.argv[2])
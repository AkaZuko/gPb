import cv2
import sys
import numpy as np

def gen_mask(width, height, color_val, img_gray):
	mask = img_gray.copy()

	for row in range(height):
		for col in range(width):
			if img_gray[row][col] == color_val:
				mask[row][col] = 255
			else:
				mask[row][col] = 0
	return mask

def masked_image(img_gray, img_color, color_val):
	height, width = img_gray.shape
	mask_ = gen_mask(width, height, color_val, img_gray)
	masked_img = cv2.bitwise_and(img_color, img_color, mask = mask_)
	return np.asarray(masked_img)

def edge_detection(img):
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

	final = cv2.Canny(im_out, 300, 400)
	return np.asarray(final)

def get_control_points(points_list):
	
	# find the RHS Matrix value say V

	# to compute M^-1 use temp = np.linalg.inv(M)

	# result ie. [ [B_1], [B_2], ... , [B_n-1]]  = np.dot(temp, V)
	return 


def process(path_seg, path_original):
	img_gray = cv2.imread(path_seg, 0)
	img_color = cv2.imread(path_original)
	
	colors = {}
	
	height1, width1 = img_gray.shape
	height2, width2, channels = img_color.shape

	height = max(height1, height2)
	width = max(width1, width2)

	img_gray = cv2.resize(img_gray, (width, height), interpolation = cv2.INTER_CUBIC)
	img_color = cv2.resize(img_color, (width, height), interpolation = cv2.INTER_CUBIC)
	
	# image_wide_edges = edge_detection(img_color)
	
	for row in range(height):
		for col in range(width):
			val = img_gray[row][col]
			if val in colors:
				colors[val] += 1
			else:
				colors[val] = 1
	# final = np.zeros((height, width), dtype=np.uint8)

	for color in colors:
		masked_img = masked_image(img_gray, img_color, color)
		edges = edge_detection(masked_img)
		# masked_img = masked_image(img_gray, image_wide_edges, color)
		# print edges.shape
		# final = cv2.bitwise_xor(final, final, mask=edges)
		# final = np.bitwise_xor(final, edges)
		cv2.imshow('result', edges)
		cv2.waitKey(0)
process(sys.argv[1], sys.argv[2])
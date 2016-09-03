import cv2
import sys
import math
import numpy as np

def get_boundary_points(img):
	print '*' * 50
	print 'get_boundary_points called'
	print '*' * 50
	print

	points = []

	height, width = img.shape[:2]

	posx = [-1,  0,  1, -1, 1, -1, 0, 1]
	posy = [-1, -1, -1,  0, 0,  1, 1, 1]

	points_dict = {}
	for row in xrange(height):
		for col in xrange(width):
			if img[row][col] == 255:
				
				stack = []
				stack.append([row, col])
				while stack != []:
					r,c = stack.pop()
					state = str(r) + " " + str(c)	
					points_dict[state] = 1
					for i in range(8):
						next_r = r + posy[i]
						next_c = c + posx[i]
						if next_c >=0 and next_c < width and next_r>=0 and next_r < height:
							next_state = str(next_r) + " " + str(next_c)
							if img[next_r][next_c] == 255 and next_state not in points_dict:
								stack.append([next_r, next_c])
	b_points = map( lambda x : map(int, x.split() ), points_dict.keys() )
	for i in b_points:
		points.append(i)
	
	return points
	# print points

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

	m10 = 0.0 
	m01 = 0.0 
	m11 = 0.0 
	m20 = 0.0 
	m02 = 0.0 

	for i in points:
		m10 = (i[0]**1)*(i[1]**0)
		m01 = (i[0]**0)*(i[1]**1)
		m11 = (i[0]**1)*(i[1]**1)
		m20 = (i[0]**2)*(i[1]**0)
		m02 = (i[0]**0)*(i[1]**2)

	m10 /= (N * 1.0 ) 
	m01 /= (N * 1.0 ) 
	m11 /= (N * 1.0 ) 
	m20 /= (N * 1.0 ) 
	m02 /= (N * 1.0 )
	
	points = map(lambda x : [ x[0] - m10, x[1] - m01 ], points)
	points = map(lambda x : [ x[0]*1.0 / (m20**0.5), x[1]*1.0 / (m02**0.5)], points)
	points = map(lambda x : [ ( x[0]*x[0] - x[1]*x[1] ) / (2**0.5), ( x[0]*x[0] + x[1]*x[1] ) / (2**0.5)], points) 
	points = map(lambda x : [ x[0]*1.0 / (m20**0.5), x[1]*1.0 / (m02**0.5)], points)

	return points


temp_img = cv2.imread(sys.argv[1], 0)
height, width = temp_img.shape[:2]
points1 = make_invariant( get_boundary_points( temp_img ) )

temp_img2 = cv2.imread(sys.argv[2], 0)
height, width = temp_img2.shape[:2]
points2 = make_invariant( get_boundary_points( temp_img2 ) )

same = True

for i in range(len(points1)):
	if points1[i] not in points2[i]:
		same = False
		break

print same
# ll = gen_image( make_invariant( get_boundary_points( temp_img ) ), height, width)
# cv2.imshow('AKA',ll)
# cv2.waitKey(0)
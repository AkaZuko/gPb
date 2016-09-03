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

	points = map(lambda x : np.complex(x[0], x[1]), points)
	FD = np.fft.fft(points)
	Z0 = FD[0]
	Z1 = FD[1]
	# translational invariance
	FD = FD[2:]

	# scalar invariance
	FD = map(lambda x : x / abs(Z0), FD)

	# rotational invariance
	
	print FD
	return FD

	# for i in range(N):
	# 	prev_ = FD[(i-1)%N]
	# 	next_ = FD[(i+1)%N]
	# 	angle = ( 2 * math.pi * i )/N
	# 	cmplx1 = np.complex(math.cos(angle), math.sin(angle))
	# 	cmplx2 = cmplx1.conjugate()
	# 	u_i = prev_*cmplx2 + next_*cmplx1
	# 	# final_points.append([u_i.real, u_i.imag])
	# 	final_points.append(u_i)


	# phi = math.atan( final_points[1].imag / final_points[1].real )
	# cmplx3 = np.complex(math.cos(phi), math.sin(phi))

	# final_points = map( lambda x : [ (x*cmplx3).real, (x*cmplx3).imag], final_points)	

	# print final_points
	# return final_points
	# return IFD


temp_img = cv2.imread(sys.argv[1], 0)
height, width = temp_img.shape[:2]
ll = gen_image( make_invariant( get_boundary_points( temp_img ) ), height, width)
cv2.imshow('AKA',ll)
cv2.waitKey(0)
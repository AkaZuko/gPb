import cv2
import sys
# import math
import numpy as np
# from collections import defaultdict

class ImageSegmentation():
	def __init__(self, path=None):
		if path:
			self.input_image = cv2.imread(path)
			# self.input_image[:,:,0] = cv2.equalizeHist(self.input_image[:,:,0])
			# self.input_image[:,:,1] = cv2.equalizeHist(self.input_image[:,:,1])
			# self.input_image[:,:,2] = cv2.equalizeHist(self.input_image[:,:,2])
			self.input_image = cv2.resize(self.input_image, (400, 400), interpolation = cv2.INTER_CUBIC)
			self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2LAB)
			# cv2.imshow("masked_2", self.input_image)
			# cv2.waitKey(0)
	
	def set_input_image(self, path):
		self.input_image = cv2.imread(path)
		self.input_image = cv2.resize(self.input_image, (50, 50), interpolation = cv2.INTER_CUBIC)
		self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2LAB)

	
	def build_filters(self):
		filters = []
		ksize = 31
		for theta in np.arange(0, np.pi, np.pi / 16):
			kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
			kern /= 1.5*kern.sum()
			filters.append(kern)
		return filters

	def build_texture_map():
		self.texture_data = [ [ [ 0.0 for ii in range(16) ] for x in range(width)] for y in range(height)]
		self.texture_data = np.asarray(self.texture_data)
		for i,filters in enumerate(self.build_filters()):
			self.texture_data[:,:,i] = cv2.filter2D(self.input_image, cv2.CV_8UC3, kern)

	def segmentation(self):
		height, width, channels = self.input_image.shape
		self.transformed = [ [ [ 0.0 for ii in range(3) ] for x in range(width)] for y in range(height)]
		self.transformed_dir = [ [ [ 0.0 for ii in range(3) ] for x in range(width)] for y in range(height)]

		for row in range(height):
			print (row+1)
			for col in range(width):


				val = [0.0, 0.0, 0.0]
				direction = [0.0, 0.0, 0.0]

				for angle in range(0, 360, 45):
					circle_img_1 = np.zeros((height,width), np.uint8)
					circle_img_2 = np.zeros((height,width), np.uint8)
			
					cv2.ellipse(circle_img_1, (col, row), (2,2), 0, (angle), (angle + 180), (255,255,255), thickness = -1)
					cv2.ellipse(circle_img_2, (col, row), (2,2), 0, (angle+180), (angle + 180 + 180), (255,255,255), thickness = -1)

					bin_count = 128

					hist_1_channel0 = cv2.calcHist([self.input_image], [0], circle_img_1, [bin_count], [0,bin_count])
					hist_2_channel0 = cv2.calcHist([self.input_image], [0], circle_img_2, [bin_count], [0,bin_count])

					hist_1_channel1 = cv2.calcHist([self.input_image], [1], circle_img_1, [bin_count], [0,bin_count])
					hist_2_channel1 = cv2.calcHist([self.input_image], [1], circle_img_2, [bin_count], [0,bin_count])

					hist_1_channel2 = cv2.calcHist([self.input_image], [2], circle_img_1, [bin_count], [0,bin_count])
					hist_2_channel2 = cv2.calcHist([self.input_image], [2], circle_img_2, [bin_count], [0,bin_count])

					temp = [0.0, 0.0, 0.0]
					for i in range(bin_count):
						if (hist_1_channel0[i][0] + hist_2_channel0[i][0] != 0):
							temp[0] += ((hist_1_channel0[i][0] - hist_2_channel0[i][0])**2)/(hist_1_channel0[i][0] + hist_2_channel0[i][0])
						else:
							temp[0] += 0.0

						if (hist_1_channel1[i][0] + hist_2_channel1[i][0] != 0):
							temp[1] += ((hist_1_channel1[i][0] - hist_2_channel1[i][0])**2)/(hist_1_channel1[i][0] + hist_2_channel1[i][0])
						else:
							temp[1] += 0.0

						if (hist_1_channel2[i][0] + hist_2_channel2[i][0] != 0):
							temp[2] += ((hist_1_channel2[i][0] - hist_2_channel2[i][0])**2)/(hist_1_channel2[i][0] + hist_2_channel2[i][0])
						else:
							temp[2] += 0.0

					temp[0] *= 0.5
					temp[1] *= 0.5
					temp[2] *= 0.5

					if(temp[0] > val[0]):
						val[0] = temp[0]
						direction[0] = angle

					if(temp[1] > val[1]):
						val[1] = temp[1]
						direction[1] = angle

					if(temp[2] > val[2]):
						val[2] = temp[2]
						direction[2] = angle

				self.transformed[row][col][0] = val[0]
				self.transformed[row][col][1] = val[1]
				self.transformed[row][col][2] = val[2]

				self.transformed_dir[row][col][0] = direction[0]
				self.transformed_dir[row][col][1] = direction[1]
				self.transformed_dir[row][col][2] = direction[2]

		ref = {0 : [[0,-1],[0,1]], 45 : [[-1,1],[1,-1]] , 90 : [[-1,0],[1,0]], 135 : [[-1,-1],[1,1]], 180 : [[0,1],[0,-1]], 225 : [[1,-1],[-1,1]], 270 : [[1,0],[-1,0]], 315 : [[1,1],[-1,-1]]}
		for channel in range(3):
			for row in range(height):
				for col in range(width):
					point1_x = row + ref[self.transformed_dir[row][col][channel]][0][0]
					point2_x = row + ref[self.transformed_dir[row][col][channel]][1][0]

					point1_y = col + ref[self.transformed_dir[row][col][channel]][0][1]
					point2_y = col + ref[self.transformed_dir[row][col][channel]][1][1]

					condition1 = True
					condition2 = True

					if(point1_x >=0 and point1_y>=0 and point1_x <row and point1_y<col and self.transformed_dir[point1_x][point1_y][channel] >= self.transformed_dir[row][col][channel]):
						condition1 = False
					if(point2_x >=0 and point2_y>=0 and point2_x <row and point2_y<col and self.transformed_dir[point2_x][point2_y][channel] >= self.transformed_dir[row][col][channel]):
						condition2 = False

					if(condition1 and condition2):
						self.transformed[row][col][channel] = self.transformed[row][col][channel]
					else:
						self.transformed[row][col][channel] = 0

		te = np.asarray(self.transformed)
		te = 255*te

		cv2.imwrite('transformed_L.png', te[:,:,0])
		cv2.imwrite('transformed_A.png', te[:,:,1])
		cv2.imwrite('transformed_B.png', te[:,:,2])

		cv2.imwrite('tranformed_con.png', te)


testing = ImageSegmentation(sys.argv[1])
testing.segmentation()
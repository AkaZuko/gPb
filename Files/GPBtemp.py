import cv2
import sys
import math
import numpy as np
from collections import defaultdict

class ImageSegmentation():
	def __init__(self, path=None):
		if path:
			self.input_image = cv2.imread(path)
			# self.input_image[:,:,0] = cv2.equalizeHist(self.input_image[:,:,0])
			# self.input_image[:,:,1] = cv2.equalizeHist(self.input_image[:,:,1])
			# self.input_image[:,:,2] = cv2.equalizeHist(self.input_image[:,:,2])
			self.input_image = cv2.resize(self.input_image, (100, 100), interpolation = cv2.INTER_CUBIC)
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
		# self.transformed_L = [ [ 0.0 for x in range(width)] for y in range(height)]
		# self.transformed_A = [ [ 0.0 for x in range(width)] for y in range(height)]
		# self.transformed_B = [ [ 0.0 for x in range(width)] for y in range(height)]

		for row in range(height):
			print (row+1)
			for col in range(width):


				val = [0.0, 0.0, 0.0]

				for i in range(0, 360, 45):
					circle_img_1 = np.zeros((height,width), np.uint8)
					circle_img_2 = np.zeros((height,width), np.uint8)
			
					cv2.ellipse(circle_img_1, (col, row), (2,2), 0, (i), (i + 180), (255,255,255), thickness = -1)
					cv2.ellipse(circle_img_2, (col, row), (2,2), 0, (i+180), (i + 180 + 180), (255,255,255), thickness = -1)
			
					bin_count = 128

					hist_1_channel0 = cv2.calcHist([self.input_image], [0], circle_img_1, [bin_count], [0,bin_count])
					hist_2_channel0 = cv2.calcHist([self.input_image], [0], circle_img_2, [bin_count], [0,bin_count])

					hist_1_channel1 = cv2.calcHist([self.input_image], [1], circle_img_1, [bin_count], [0,bin_count])
					hist_2_channel1 = cv2.calcHist([self.input_image], [1], circle_img_2, [bin_count], [0,bin_count])

					hist_1_channel2 = cv2.calcHist([self.input_image], [2], circle_img_1, [bin_count], [0,bin_count])
					hist_2_channel2 = cv2.calcHist([self.input_image], [2], circle_img_2, [bin_count], [0,bin_count])

					# hist_1_channel0[np.isnan(hist_1_channel0)] = 0.0
					# hist_1_channel1[np.isnan(hist_1_channel1)] = 0.0
					# hist_1_channel2[np.isnan(hist_1_channel2)] = 0.0
					# hist_2_channel0[np.isnan(hist_2_channel0)] = 0.0
					# hist_2_channel1[np.isnan(hist_2_channel1)] = 0.0
					# hist_2_channel2[np.isnan(hist_2_channel2)] = 0.0
					
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

					val[0] = max(temp[0], val[0])
					val[1] = max(temp[1], val[1])
					val[2] = max(temp[2], val[2])

				# self.transformed_L[row][col] = val[0]
				# self.transformed_A[row][col] = val[1]
				# self.transformed_B[row][col] = val[2]

				self.transformed[row][col][0] = val[0]
				self.transformed[row][col][1] = val[1]
				self.transformed[row][col][2] = val[2]

		# , dtype = np.float16
		# , dtype = np.float16
		# , dtype = np.float16

		

		
		te = np.asarray(self.transformed)
		# if(te[:,:,0].max()!=0):
		# 	te[:,:,0] = 255*(te[:,:,0]/(te[:,:,0].max()*1.0))
		# if(te[:,:,1].max()!=0):
		# 	te[:,:,1] = 255*(te[:,:,1]/(te[:,:,1].max()*1.0))
		# if(te[:,:,2].max()!=0):
		# 	te[:,:,2] = 255*(te[:,:,2]/(te[:,:,2].max()*1.0))
		te = 255*te
		
		with open('tata.txt','w') as f:
			for row in range(height):
				daa = ''
				for col in range(width):
					daa+= str(te[row][col][0]) + ' '
				f.write(daa + '\n')

		cv2.imwrite('transformed_L.png', np.asarray(te[:,:,0]))
		cv2.imwrite('transformed_A.png', np.asarray(te[:,:,1]))
		cv2.imwrite('transformed_B.png', np.asarray(te[:,:,2]))

		cv2.imwrite('tranformed_con.png', te)


testing = ImageSegmentation(sys.argv[1])
testing.segmentation()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cv2
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb


image = cv2.imread('Selection_008.png',0)

# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(3))

# remove artifacts connected to image border
cleared = bw.copy()
clear_border(cleared)

# label image regions
label_image = label(cleared)
borders = np.logical_xor(bw, cleared)
label_image[borders] = -1
image_label_overlay = label2rgb(label_image, image=image)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(image_label_overlay)

regions = 0
for region in regionprops(label_image):
	regions += 1
	# skip small images
	#if region.area < 100:
	#    continue
	# draw rectangle around segmented coins
	minr, minc, maxr, maxc = region.bbox
	rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=True, edgecolor='red', linewidth=2)
	ax.add_patch(rect)

print regions	
plt.show()

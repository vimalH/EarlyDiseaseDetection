import math
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data,io,exposure
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square, opening,disk
from skimage.measure import regionprops, perimeter
from skimage.color import label2rgb, rgb2hsv,rgb2lab
from skimage.util.shape import view_as_blocks
from skimage.transform import resize
from sklearn.externals import joblib
from skimage.filters import threshold_adaptive
from sklearn import tree

import os



#clf = joblib.load('fungalDetectModel.pkl')

def build_feature_file():
	'''
	takes a directory with images and builds a file with feature descriptors and labels
	'''
	import os, os.path
	image_urls = []
	image_dir = os.path.join(os.getcwd(),'images','downey mildew')

	path = image_dir
	valid_images = [".jpg",".gif",".png",".tga"]
	with open('datafile.csv','a') as data_file:
		for f in os.listdir(path):
			print(f)
			img_path = os.path.join(image_dir,f)
			try:
			    fd = ','.join([str(i) for i in process_image(img_path)])
			except:
			    continue
			label = 2
			fd += ','+str(label)+'\n'
			data_file.write(fd)

def save_model(clf):
	joblib.dump(clf, 'fungalDetectModel.pkl')



def learn():
	from sklearn import svm
	d = np.genfromtxt('datafile.csv',delimiter=',')

	X = d[:,:d.shape[1]-1]
	y = d[:,d.shape[1]-1]
	clf = svm.SVC()
	#clf = tree.DecisionTreeClassifier()

	clf.fit(X,y)
	#save model
	save_model(clf)


def predict(image_path):

	mapper = {1: 'Anthracnose',2: 'Downey Mildew',3:'Early Blight',4:'Leaf Blight',5:'Leaf Rust',6:'Leaf Spot',7:'Normal',8:'Powdery Mildew'}
	clf = joblib.load('fungalDetectModel.pkl')
	a = process_image(image_path)
	# print(len(a))

	#print(int(clf.predict([a])[0]))
	return mapper.get(int(clf.predict([a])[0]))

	# return mapper.get()


def extract_features(regionprops):
	'''
	function to extract features from a given region, return a list of features
	'''
	avg_mean_intensity = np.average([region.mean_intensity for region in regionprops])
	avg_min_intensity = np.average([region.min_intensity for region in regionprops])
	avg_max_intensity = np.average([region.max_intensity for region in regionprops])
	avg_moments = np.average([max(region.moments_hu) for region in regionprops])
	average_area = np.average([region.filled_area for region in regionprops])
	max_area = max(regionprops, key=lambda region: region.filled_area).filled_area
	average_perimeter = np.average([region.perimeter for region in regionprops])
	average_extent = np.average([region.extent for region in regionprops])
	average_eccentricity = np.average([region.eccentricity for region in regionprops])
	average_equivalent_diameter = np.average([region.equivalent_diameter for region in regionprops])
	return [avg_mean_intensity,avg_min_intensity,avg_max_intensity,avg_moments,average_eccentricity,average_perimeter,max_area,average_area,average_equivalent_diameter]

def process_image(image_path):
	# image = io.imread(image_path,as_grey=True)
	image = rgb2lab(io.imread(image_path))[:,:,1]	#contrast stretching
	p2, p98 = np.percentile(image, (2, 98))
	img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
	# io.imshow(img_rescale)
	# plt.savefig("processed.png")
	# fig, (ax1,ax_hist) = plt.subplots(ncols=1, nrows=2, figsize=(6, 6))
	# apply threshold
	threshold = threshold_otsu(img_rescale)
	bw = opening(image > threshold,square(3))


	fig, ax = plt.subplots(figsize=(4, 3))
	ax.imshow(bw, cmap=plt.cm.gray, interpolation='nearest')
	ax.axis('off')
	ax.set_title('Binary Image')
	#plt.show()

	labeled_image = label(bw)
	image_label_overlay = label2rgb(labeled_image, image=image)
	
	regions = regionprops(labeled_image,intensity_image=image)


	# io.imsave('processed.png', image_label_overlay)
	# plt.savefig("processed.png")
	# ax_hist.plot(hist[1][:-1], hist[0], lw=2)
	ax.imshow(image_label_overlay, cmap=plt.cm.gray, interpolation='nearest')
	#plt.show()


	return extract_features(regions)


if __name__ == '__main__':

	#build_feature_file()
	learn()
	print(predict('images/anthracnose/2.jpg'))




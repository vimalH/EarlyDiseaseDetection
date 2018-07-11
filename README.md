# EarlyDiseaseDetection


Machine learning based web application for automatic detection and prediction of leaf and diseases in grapes.


Image Preprocessing

Image is pre-processed to improve the image Pre-processing includes color conversion, histogram, and histogram equalization. Color conversion and histogram equalization is used to improve the quality and clarity images. Grayscale images are easy to process in any application because they have only intensity values. The histogram equalization enhances the contrast of images by transforming the intensity values. 

Image Segmentation

	Image segmentation is typically used to locate objects and boundaries.
	 Segmentation partitions an image into distinct regions containing each pixel with similar attributes.
	There are few segmentation algorithms like Otsu method, K-means Algorithm.
	They produce an optimal matrix to redefine the position of infected part in fruit/leaf.

Thresholding and Clustering

	The simplest method of image segmentation.
	Based on a threshold value, a gray scale image is turned into a binary image.
	Replace each pixel in an image with a black pixel if the image intensity I i,j is less than some fixed constant T or a white pixel if the image intensity is greater than that constant

Feature Extraction

The feature extraction and representation technique are used to convert the segment objects into representations that describe their main features and attributes. Extract feature from a given region, return a list of features such as region_count, average_area, max_area, average_perimeter,average_euler_number, average_eccentricity, average_equivalent_ diameter

SVM Classifier

SVMs (Support Vector Machines) are a useful technique for data classification. Classification task usually involves separating data into training and testing sets. Each instance in the training set contains one \target value" (i.e. the class labels) and several attributes" (i.e. the features or observed variables). The goal of SVM is to produce a model (based on the training data) which predicts the target values of the test data given only the test data attributes. A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labelled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples.

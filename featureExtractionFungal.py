import math
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data,io,exposure
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, square, opening,disk
from skimage.measure import regionprops, perimeter
from skimage.color import label2rgb, rgb2hsv
from skimage.util.shape import view_as_blocks
from skimage.transform import resize
from sklearn.externals import joblib
from skimage.segmentation import clear_border

import os




def extract_features(regionprops):
    '''
    function to extract features from a given region, return a list of features
    '''

    region_count = len(regionprops)
    average_area = np.average([region.filled_area for region in regionprops])
    max_area = max(regionprops, key=lambda region: region.filled_area).filled_area
    average_perimeter = np.average([region.perimeter for region in regionprops])
    average_euler_number = np.average([region.euler_number for region in regionprops])
    average_eccentricity = np.average([region.eccentricity for region in regionprops])
    average_equivalent_diameter = np.average([region.equivalent_diameter for region in regionprops])
    return [region_count,average_eccentricity,average_perimeter,max_area,average_area,average_equivalent_diameter,average_euler_number]

def get_fungal_features(image_path):
    # image = rgb2hsv(io.imread(image_path))[:,:,1]
    image = io.imread(image_path,as_grey=True) #read and convert to grey scale
    io.imshow(image)
    plt.show()
    #contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    io.imshow(img_rescale)
    plt.show()
    fig, ax1 = plt.subplots(ncols=1, nrows=1)
    #histogram
    hist = np.histogram(image)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax1.axis('off')
    ax2.plot(hist[1][:-1], hist[0], lw=2)
    ax2.set_title('histogram of grey values')
    plt.show()
    # apply threshold(otsu thresholding)
    thresh = threshold_otsu(img_rescale)

    #morphological operation(opening) removes small spots
    bw = opening(image > thresh, square(3))
    io.imshow(bw)
    plt.show()


    #region labeling
    label_image = label(bw)
    image_label_overlay = label2rgb(label_image, image=image)
    io.imshow(image_label_overlay)
    plt.show()
    # io.imsave('processed.png', image_label_overlay)
    # plt.savefig("processed.png")

    # ax.imshow(image_label_overlay, cmap=plt.cm.gray, interpolation='nearest')
    # plt.show()
    # plt.savefig('processed.png')

    return extract_features(regionprops(label_image))



if __name__ == '__main__':
    # burintild_feature_file()
    print(get_fungal_features('d2.jpg'))


# This code was written for a Kaggle challenge to identify cells given image from a microscope.
# Our challenge was to practice image processing and


import numpy as np
import pandas as pd
import pathlib
import imageio
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy import ndimage


# This method run length encodes the locations of the cell nuclei in a mask. Run length encoding
# simplifies the location of the mask into data that we can easily return as a dataframe for submission
# to the competition.
def run_length_encoding(mask):
    
    # The .T method takes the transpose of the mask and then finds the indices where the objects
    # are in the mask.
    dots = np.where(mask.T == 1)[0]
    run_lengths = []
    prev = -2

    # We iterate through the indices where there are objects and compute the run lengths that we return.
    for value in dots:
        if (value>prev+1):
            run_lengths.extend((value+1, 0))
        run_lengths[-1] += 1
        prev = value

    # This loop iterates through and conjoins the run lengths into a single string.
    return " ".join([str(i) for i in run_lengths])


# This function will scan the image and get the data on the location of cells in the picture.
def scan_image(img_path):
    
    # This dataframe will store the data we get from the current image.
    image_data = pd.DataFrame()

    # The image_id is the name of the image that we use to identify it.
    image_id = img_path.parts[-3]
    
    # We read the image at the specified path and grayscale it.
    image = imageio.imread(str(img_path))
    image = rgb2gray(image)
    
    # We use the threshold_otsu function to be able to have the threshold value be based on
    # the histogram of color values and get the central value that is not defined by the mean
    # or the median. It will be defined by the point where the area on both sides will be equal.
    threshold_value = threshold_otsu(image)
    mask = np.where(image > threshold_value, 1, 0)
    
    # ndimage.label is a built-in function of scipy that determines independent objects and
    # labels them as different numbers, which is basically the image segmentation that we
    # wanted to do.
    labels, number_of_objects = ndimage.label(mask)

    # For each label that we have created, we create, analyze, and encode a mask corresponding to
    # each independent nucleus and then add them to a dataframe for final submission.
    for label in range(1, number_of_objects+1):
        
        # We isolate the label and set every pixel outside the label to 0, and the pixel in the label to 1.
        label_mask = np.where(labels == label, 1, 0)
        
        # This if statement ensures that the size of the nucleus is at least 10 pixels, which
        # prevents us from looking at background noise and random dots in the picture and classifying
        # them as nuclei. Once we find a mask, we encode it and add to the data for the current image.
        if label_mask.sum() > 10:
            rle = run_length_encoding(label_mask)
            series = pd.Series({'ImageId': image_id, 'EncodedPixels': rle})
            image_data = image_data.append(series, ignore_index=True)

    return image_data


# We create a generator for all of the image paths in our image folder that we want to analyze.
img_paths = pathlib.Path('../input/images').glob('*/images/*.png')


# This blank dataframe will store our results so that we can convert them back into a csv.
results = pd.DataFrame()


# For each image path in the list, we scan the image and return the cells and their run length encoded locations and
# then we add this data to our results.
for im_path in list(img_paths)
    data = scan_image(im_path)
    results = results.append(data, ignore_index = True)


# We organize our dataframe and then convert it back into a csv for submission.
column_titles = ['ImageId', 'EncodedPixels']
results = results.reindex(columns = column_titles)
results.to_csv('output.csv', index = False)

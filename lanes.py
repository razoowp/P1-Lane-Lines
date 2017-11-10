#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import cv2

from moviepy.editor import VideoFileClip
from IPython.display import HTML

import helper

# %matplotlib inline

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

# image = helper.grayscale(image)
#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

def preprocessing(image):
    
    #grayscale
    gray = helper.grayscale(image)
    #plt.imshow(gray, cmap="grayJ

    #gaussain blur
    kernel_size = 5
    blur_gray = helper.gaussian_blur(gray, kernel_size)
    #plt.imshow(blur_gray, cmap="gray")

    #Canny
    low_threshold = 50
    high_threshold = 150
    edges = helper.canny(blur_gray, high_threshold,low_threshold)
    #plt.imshow(edges, cmap="gray")

    #mask
    v1 = (50,539)
    v2 = (470,320)
    v3 = (500,320)
    v4 = (900,539)

    vertices = np.array([[v1,v2,v3,v4]],dtype=np.int32)
    masked_edges = helper.region_of_interest(edges,vertices)
    # plt.imshow(masked_edges, cmap="gray")
    # plt.show()

    #hough lines
    rho = 1
    theta = np.pi/180
    threshold = 30
    min_line_length = 5
    max_line_gap = 10

    line_image = helper.hough_lines(masked_edges,rho,theta,threshold,min_line_length,max_line_gap)
    # plt.imshow(line_image, cmap="gray")
    # plt.show()

    #region of interest for the lines
    masked_line_image = helper.region_of_interest(line_image, vertices)
    # plt.imshow(masked_line_image, cmap = "gray")
    # plt.show()

    #weighted image
    weighted_image = helper.weighted_img(masked_line_image,image,0.8,0.5,0)
    # plt.imshow(weighted_image, cmap ="gray")
    # plt.show()
    
    return weighted_image
    
all_images = os.listdir("test_images/")

# for image_name in all_images:
image = cv2.imread('test_images/'+all_images[5])
#     print (image_name)
processed_image = preprocessing(image)
# cv2.imwrite("test_"+image_name,processed_image)
    #input("Enter to continue")

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = preprocessing(image)

    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(preprocessing) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
white_clip.show
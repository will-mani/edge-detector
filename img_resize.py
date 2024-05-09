import numpy as np
import cv2

factor = 15
image = cv2.imread('eagle.jpg')
image_height, image_width, imgage_channels = image.shape
resized_height, resized_width = image_height * factor, image_width * factor
resized_image = np.zeros((resized_height, resized_width, imgage_channels), dtype=np.uint8)

row_coors = np.arange(resized_height * resized_width) / (resized_width * factor) # 1D array of floats
row_coors = np.array(row_coors, dtype=np.uint) # 1D array of ints

col_coors = np.zeros((resized_height, resized_width)) + np.arange(resized_width) / factor # resized_height by resized_width array of floats
col_coors = np.reshape(np.array(col_coors, dtype=np.uint), row_coors.shape) # 1D array of ints

resize_coors = row_coors, col_coors
resized_image[::] = np.reshape(image[resize_coors], resized_image.shape)

print(image.shape)
print(resized_image.shape)

cv2.imshow('Original', image)
cv2.imshow('Resized', resized_image)
cv2.waitKey(0)

# cv2.imwrite('eagle15X.png', resized_image)
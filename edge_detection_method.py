import numpy as np
import cv2

threshold = 30
image = cv2.imread('eagle.jpg')
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

grayscale_type_float = np.array(grayscale, dtype = np.float16)
image_height, image_width = grayscale.shape

horizontal_differences = grayscale_type_float[:image_height-1,:] - grayscale_type_float[1:,:]
horizontal_edges = np.array(np.abs(horizontal_differences), dtype = np.uint8)
horizontal_edges_coors = np.where(horizontal_edges >= threshold)[0], np.where(horizontal_edges >= threshold)[1]

vertical_differences = grayscale_type_float[:,:image_width-1] - grayscale_type_float[:,1:]
vertical_edges = np.array(np.abs(vertical_differences), dtype=np.uint8)
vertical_edges_coors = np.where(vertical_edges >= threshold)[0], np.where(vertical_edges >= threshold)[1]

detected_edges_image = np.zeros(grayscale.shape, dtype=np.uint8)
detected_edges_image[horizontal_edges_coors] = 255
detected_edges_image[vertical_edges_coors] = 255

# cv2.imshow('Original', grayscale)
cv2.imshow('Edges', detected_edges_image)
cv2.waitKey(0)
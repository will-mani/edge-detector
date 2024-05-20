import numpy as np
import cv2

### Instructions:
### First, set the image path and factor defined below the edge_detector class.
### Then, run the program and set the three trackbars (kernal size, threshold, and factor) to the desired values.
### Experiment and have fun!

class edge_detector():
    def __init__(self, original_image): 
        self.original_image = original_image
        self.original_grayscale = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.kernal_size = 1
        self.blurred_grayscale = self.original_grayscale.copy()
        self.threshold = 0
        self.detected_horizontal_edges = np.ones(self.blurred_grayscale.shape, dtype=np.uint8) * 255
        self.detected_vertical_edges = np.ones(self.blurred_grayscale.shape, dtype=np.uint8) * 255
        self.detected_edges = self.detected_horizontal_edges + self.detected_vertical_edges
        self.factor = 1
        self.resized_edges = np.zeros(self.detected_edges.shape, dtype=np.uint8)
        

    def blur_grayscale(self, kernal_size):
        self.kernal_size = kernal_size
        self.blurred_grayscale = cv2.blur(self.original_grayscale, (kernal_size, kernal_size))

        self.detect_edges(self.threshold)


    def detect_edges(self, threshold):
        self.threshold = threshold

        float_blurred_grayscale = np.array(self.blurred_grayscale, dtype = np.float16)
        image_height, image_width = float_blurred_grayscale.shape

        horizontal_differences = float_blurred_grayscale[:image_height-1,:] - float_blurred_grayscale[1:,:]
        horizontal_edges = np.array(np.abs(horizontal_differences), dtype = np.uint8)
        horizontal_edges_coors = np.where(horizontal_edges >= threshold)

        vertical_differences = float_blurred_grayscale[:,:image_width-1] - float_blurred_grayscale[:,1:]
        vertical_edges = np.array(np.abs(vertical_differences), dtype=np.uint8)
        vertical_edges_coors = np.where(vertical_edges >= threshold)

        self.detected_horizontal_edges = np.zeros(self.blurred_grayscale.shape, dtype=np.uint8)
        self.detected_horizontal_edges[horizontal_edges_coors] = 255
        self.detected_vertical_edges = np.zeros(self.blurred_grayscale.shape, dtype=np.uint8)
        self.detected_vertical_edges[vertical_edges_coors] = 255

        self.detected_edges = self.detected_horizontal_edges + self.detected_vertical_edges

        self.resize_edges(self.factor)


    def resize_edges(self, factor):
        self.factor = factor

        original_height, original_width = self.original_grayscale.shape
        resized_height, resized_width = original_height * factor, original_width * factor

        resized_horizontal_edges = np.zeros((original_height, original_width, factor), dtype=np.uint8) + self.detected_horizontal_edges.reshape((original_height, original_width, 1))
        resized_horizontal_edges = resized_horizontal_edges.reshape((original_height, resized_width))
        resized_vertical_edges = np.zeros((original_height, factor, original_width), dtype=np.uint8) + self.detected_vertical_edges.reshape((original_height, 1, original_width))
        resized_vertical_edges = resized_vertical_edges.reshape((resized_height, original_width))

        horizontal_edges_image = np.zeros((resized_height, resized_width), dtype=np.uint8)
        horizontal_edges_image[::factor,:] = resized_horizontal_edges
        vertical_edges_image = np.zeros((resized_height, resized_width), dtype=np.uint8)
        vertical_edges_image[:, ::factor] = resized_vertical_edges
        resized_edges_image = horizontal_edges_image + vertical_edges_image # Need to avoid covering horizontal edges (white pixel values) with vertical 'non-edges' (black pixel values). Hence, the addition
        
        display_edges_image = resized_edges_image.copy()

        ### The cv2 display might take up the whole screen unless the code snippet below for getting back to the original size is uncommented
        display_factor = factor
        while display_factor > 1:
            display_factor -= 1
            display_width, display_height = original_width * display_factor, original_height * display_factor
            display_edges_image = cv2.resize(display_edges_image, (display_width, display_height))

        self.resized_edges = cv2.bitwise_not(display_edges_image) # cv2.bitwise_not inverts the image color



image_path = "eagle.jpg"
factor = 1
image = cv2.imread(image_path)
image = cv2.resize(image, (0,0), fx=factor, fy=factor)
detector = edge_detector(image)

cv2.namedWindow('Blurred Grayscale') 
cv2.createTrackbar('Kernal Size', 'Blurred Grayscale', 1, 15, detector.blur_grayscale) # The trackbar is initially set to 1. The minimum (0 by default) is not set here
cv2.setTrackbarMin('Kernal Size', 'Blurred Grayscale', 1) # The trackbar minimum is set to 0 without this line

cv2.namedWindow('Edges') 
cv2.createTrackbar('Threshold', 'Edges', 0, 255, detector.detect_edges)

cv2.namedWindow('Resized Edges') 
cv2.createTrackbar('Factor', 'Resized Edges', 1, 15, detector.resize_edges)
cv2.setTrackbarMin('Factor', 'Resized Edges', 1)

while(True): 
    cv2.imshow('Blurred Grayscale', detector.blurred_grayscale) 
    cv2.imshow('Edges', detector.detected_edges)
    cv2.imshow('Resized Edges', detector.resized_edges) 
  
    key = cv2.waitKey(1)

    if key == 27: # esc(ape) key
        break

cv2.destroyAllWindows() 
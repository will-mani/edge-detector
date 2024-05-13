import numpy as np
import cv2

### Instructions:
### Set the image path by modifyig the image_path variable below the class definition.
### Once you run the program, use the trackbars in the two windows that pop up to
### set the desired values for kernal_size (for image blurring), 
### factor (x times bigger than the original), and threshold (for determining edges).
### Press the enter key once you have set all three values. To exit the program, hit the esc(ape) key.
### Have fun experimenting (and no need to be alarmed if the screen is blank at first; see last line of __init__ method to understand why)!

class edge_detector():
    def __init__(self, original_image): 
        self.original_image = original_image
        self.original_grayscale = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.kernal_size = 0
        self.blurred_grayscale = self.original_grayscale
        self.factor = 1
        self.resized_blurred_grayscale = self.blurred_grayscale
        self.threshold = 0
        self.edges_image = np.zeros(self.blurred_grayscale.shape, dtype=np.uint8)
        

    def blur_grayscale(self, kernal_size, factor, threshold):
        self.kernal_size = kernal_size
    
        if kernal_size < 1:
            self.blurred_grayscale = self.original_grayscale
            print("Original grayscale; no blurring involved")
        else:
            self.blurred_grayscale = cv2.blur(self.original_grayscale, (kernal_size, kernal_size))
            print("Blurred grayscale with kernal size", kernal_size, "by", kernal_size)
        
        self.resize_blurred_grayscale(factor, threshold)


    def resize_blurred_grayscale(self, factor, threshold): 
        print("Setting factor to", factor, end = ". ")
        self.factor = factor

        image_height, image_width = self.blurred_grayscale.shape
        resized_height, resized_width = image_height * factor, image_width * factor
        self.resized_blurred_grayscale = np.zeros((resized_height, resized_width), dtype=np.uint8)

        row_coors = np.arange(resized_height * resized_width) / (resized_width * factor) # 1D array of floats
        row_coors = np.array(row_coors, dtype=np.uint) # 1D array of ints

        col_coors = np.zeros((resized_height, resized_width)) + (np.arange(resized_width) / factor) # resized_height by resized_width array of floats
        col_coors = np.reshape(np.array(col_coors, dtype=np.uint), row_coors.shape) # 1D array of ints

        resize_coors = row_coors, col_coors
        self.resized_blurred_grayscale[::] = np.reshape(self.blurred_grayscale[resize_coors], (resized_height, resized_width))

        print("Now detecting edges...")
        self.detect_edges(threshold)


    def detect_edges(self, threshold):
        print("Threshold =", threshold, end = "? ")
        self.threshold = threshold

        resized_type_float = np.array(self.resized_blurred_grayscale, dtype = np.float16)
        image_height, image_width = resized_type_float.shape

        horizontal_differences = resized_type_float[:image_height-1,:] - resized_type_float[1:,:]
        horizontal_edges = np.array(np.abs(horizontal_differences), dtype = np.uint8)
        horizontal_edges_coors = np.where(horizontal_edges >= threshold)[0], np.where(horizontal_edges >= threshold)[1]

        vertical_differences = resized_type_float[:,:image_width-1] - resized_type_float[:,1:]
        vertical_edges = np.array(np.abs(vertical_differences), dtype=np.uint8)
        vertical_edges_coors = np.where(vertical_edges >= threshold)[0], np.where(vertical_edges >= threshold)[1]

        resized_edges_image = np.ones(self.resized_blurred_grayscale.shape, dtype=np.uint8) * 255 # image_height by image_width matrix full of 255's
        resized_edges_image[horizontal_edges_coors] = 0
        resized_edges_image[vertical_edges_coors] = 0

        display_edges_image = resized_edges_image.copy()

        ### The cv2 display might take up the whole screen unless the code snippet below for getting back to the original size is uncommented
        original_height, original_width = self.original_grayscale.shape
        display_factor = self.factor
        while display_factor > 1:
            display_factor -= 1
            display_width, display_height = original_width * display_factor, original_height * display_factor
            display_edges_image = cv2.resize(display_edges_image, (display_width, display_height))

        self.edges_image = display_edges_image
        
        print("Done!")

image_path = 'eagle.jpg'
image = cv2.imread(image_path)
detector = edge_detector(image)


def do_nothing(trackbar_value): 
    pass

cv2.namedWindow('Blurred Grayscale') 
cv2.createTrackbar('Kernal Size', 'Blurred Grayscale', 0, 15, do_nothing)

cv2.namedWindow('Edges') 
cv2.createTrackbar('Factor', 'Edges', 1, 15, do_nothing) # The trackbar is initially set to 1. The minimum (0 by default) is not set here
cv2.setTrackbarMin('Factor', 'Edges', 1) # The trackbar minimum is set to 0 without this line
cv2.createTrackbar('Threshold', 'Edges', 0, 255, do_nothing)

while(True): 
    cv2.imshow('Blurred Grayscale', detector.blurred_grayscale) 
    cv2.imshow('Edges', detector.edges_image) 

    kernal_size = cv2.getTrackbarPos('Kernal Size', 'Blurred Grayscale') 
    factor = cv2.getTrackbarPos('Factor', 'Edges')
    threshold = cv2.getTrackbarPos('Threshold', 'Edges')
  
    key = cv2.waitKey(1)

    if key == 13: # enter key
        if kernal_size != detector.kernal_size:
            detector.blur_grayscale(kernal_size, factor, threshold)
        elif factor != detector.factor:
            detector.resize_blurred_grayscale(factor, threshold)
        elif threshold != detector.threshold:
            detector.detect_edges(threshold)

    if key == 27: # esc(ape) key
        break

cv2.destroyAllWindows() 
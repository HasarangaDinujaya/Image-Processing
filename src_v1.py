import cv2
import numpy as np
kernal = np.ones((7, 7), np.uint8)



# Read the original color image
original_image = cv2.imread('res/original.jpg', cv2.IMREAD_COLOR)

# Read the binary template image
template = cv2.imread('res/template_1.jpg', cv2.IMREAD_GRAYSCALE)

# Convert the original image to grayscale
original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Apply binary inverse thresholding to the grayscale image
_, thresholded_image = cv2.threshold(original_image_gray, 127, 255, cv2.THRESH_BINARY_INV)
thresholded_image = cv2.medianBlur(thresholded_image, ksize= 7 )
thresholded_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernal)


# Get the width and height of the binary template
w, h = template.shape[::-1]

# Perform template matching on the thresholded image using cv2.TM_CCOEFF_NORMED
result = cv2.matchTemplate(thresholded_image, template, cv2.TM_CCOEFF_NORMED)

# Set a threshold value to identify the regions where the template matches
matching_threshold = 0.7  # Adjust this value as needed
locations = np.where(result >= matching_threshold)

# Draw rectangles on the original color image around the detected regions
for pt in zip(*locations[::-1]):
    cv2.rectangle(original_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

# Display the original image with marked regions
cv2.imshow('Detected Template', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

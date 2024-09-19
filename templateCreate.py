import cv2
import numpy as np

# Initialize global variables
start_point = None
end_point = None
selecting = False

# Mouse callback function to select an area
def select_area(event, x, y, flags, param):
    global start_point, end_point, selecting

    if event == cv2.EVENT_LBUTTONDOWN:
        # Left mouse button down: start selecting the area
        start_point = (x, y)
        selecting = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            # Mouse is moving and selection is active: show the selected rectangle
            img_copy = img_resized.copy()
            end_point = (x, y)
            cv2.rectangle(img_copy, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow('Image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        # Left mouse button released: end selection
        end_point = (x, y)
        selecting = False
        cv2.rectangle(img_resized, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow('Image', img_resized)

        # Apply threshold to the selected area
        apply_threshold_to_selection()

def apply_threshold_to_selection():
    global start_point, end_point

    if start_point and end_point:
        # Extract the selected region from the resized grayscale image
        x1, y1 = start_point
        x2, y2 = end_point
        selected_area = gray_resized[y1:y2, x1:x2]

        # Apply a binary threshold (you can change threshold value and type)
        _, thresholded = cv2.threshold(selected_area, 128, 255, cv2.THRESH_BINARY)

        # Save the thresholded image
        cv2.imwrite('thresholded_image.png', thresholded)
        cv2.imshow('Thresholded Image', thresholded)

# Load the image
img = cv2.imread(r'scr\100SCR\rs100Back.jpg')  # Replace with your image file

# Resize the image (change the width and height to desired dimensions)
width = 600  # Specify the desired width
height = 400  # Specify the desired height
img_resized = cv2.resize(img, (width, height))

# Convert the resized image to grayscale
gray_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Show the resized image and set the mouse callback
cv2.imshow('Image', img_resized)
cv2.setMouseCallback('Image', select_area)

# Wait for the user to press 'q' to quit
cv2.waitKey(0)
cv2.destroyAllWindows()

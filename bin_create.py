import cv2
import numpy as np
import matplotlib.pyplot as plt

main_image = cv2.imread(r'scr\20SCR\slllogo.jpg', cv2.IMREAD_COLOR)

# Convert the original image to grayscale
original_image_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
_,binary = cv2.threshold(original_image_gray,120,255,cv2.THRESH_BINARY_INV)


cv2.imshow('test', binary)
cv2.imwrite('test.jpg', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
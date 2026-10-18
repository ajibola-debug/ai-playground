import cv2 as cv
import numpy as np

image = cv.imread('.../Martin_Odegaard_Arsenal_2023-24_(2).jpg')
cv.imshow("Image", image)
cv.imwrite("orignal_image.jpg", image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)
cv.imwrite("greyed.jpg", gray)

threshold, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
cv.imshow("Thresh", thresh)
cv.imwrite('thres.jpg', thresh)

threshold, thresh_inv = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
cv.imshow("Thresh inv", thresh_inv)
cv.imwrite("thres_inv.jpg", thresh_inv)

adapt = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C , cv.THRESH_BINARY, 15, 3)
cv.imshow("Adative thresh", adapt)
cv.imwrite('adaptive.jpg', adapt)

cv.waitKey(0)

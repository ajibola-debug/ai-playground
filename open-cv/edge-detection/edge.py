import cv2 as cv
import numpy as np

image = cv.imread('.../batits.jpg')
cv.imshow("Batistuta", image)
cv.imwrite("original_image.jpg", image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("gray 🙄🙄🙄", gray)
cv.imwrite('grayed.jpg', gray) 

lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)
cv.imwrite('Laplacian.jpg', lap)

sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)

cv.imshow("Sobelx", sobelx)
cv.imwrite('sobel_x.jpg', sobelx)

cv.imshow("Sobely", sobely)
cv.imwrite('sobel_y.jpg', sobely)

bit = cv.bitwise_or(sobelx, sobely)
cv.imshow("Bitting", bit)
cv.imwrite("bitwised.jpg", bit)

cv.waitKey(0)

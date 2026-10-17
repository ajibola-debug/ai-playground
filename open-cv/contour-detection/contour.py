import cv2 as cv

img = cv.imread('.../Martin Odegaard Arsenal 2023-24 (8).jpg')

cv.imshow("Image", img)

(height, width) = img.shape[:2]

resize = cv.resize(img, (width//3, height//3))
cv.imwrite("orignal-iamge.jpg", resize)

gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)
cv.imwrite("grey.jpg", gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow("Blur", blur)
cv.imwrite("blur.jpg", blur)

canny = cv.Canny(resize, 125, 175)
cv.imshow("Canny", canny)
cv.imwrite("cannny.jpg", canny)

ret, tresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow("Thresh", tresh)

contours, hierachies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'Number of countour(s) found: {len(contours)}')

cv.imwrite("Ode.jpg", tresh)

cv.waitKey(0)


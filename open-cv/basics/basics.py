import cv2 as cv

img = cv.imread('images/image6.jpg')
cv.imshow('image', img)

grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('grey', grey)
cv.imwrite("gray.jpg", grey)

#Blur
blur = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('blur', blur)
cv.imwrite("blur.jpg", blur)

canny = cv.Canny(blur, 125, 175)
cv.imshow('canny', canny)
cv.imwrite("canny.jpg", canny)

dilated = cv.dilate(canny, (7, 7), iterations=3)
cv.imshow('dilated', dilated)
cv.imwrite("dilated.jpg", dilated)

enroding = cv.erode(dilated, (7, 7), iterations=3)
cv.imshow('enroding', enroding)
cv.imwrite("enrode.jpg", enroding)

#resize
resize = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('resize', resize)
cv.imwrite("resize.jpg", resize)
cv.waitKey(0)
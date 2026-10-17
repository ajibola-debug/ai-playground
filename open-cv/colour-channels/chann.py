import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("images/ferri.jpg")

cv.imshow("Image", image)
cv.imwrite('original_image.jpg', image)

b,g,r = cv.split(image)

cv.imshow("Blue", b)
cv.imwrite("blue.jpg", b)
cv.imshow("Green", g)
cv.imwrite("green.jpg", g)
cv.imshow("Red", r)
cv.imwrite("red.jpg", r)

print(image.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merge = cv.merge([b, g, r])
cv.imshow("Merged!", merge)
cv.imwrite("merged.jpg", merge)

cv.waitKey(0)

#plt.imshow(rgb)
#plt.show()

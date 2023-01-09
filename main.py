import cv2
from cv2 import dnn_superres
supres = dnn_superres.DnnSuperResImpl_create()

image = cv2.imread('sample.jpeg')

path = "LapSRN_x8.pb"
supres.readModel(path)


supres.setModel("lapsrn", 8)
result = supres.upsample(image)

cv2.imwrite("upscaled.jpeg", result)
cv2.imwrite("./upscaled.png", result)
import cv2
from cv2 import dnn_superres
super_res = dnn_superres.DnnSuperResImpl_create()

img = cv2.imread('sample.jpeg')

path = "LapSRN_x8.pb"
super_res.readModel(path)

super_res.setModel("lapsrn", 8)
out_img = super_res.upsample(img)

cv2.imwrite("upscaled.jpeg", out_img)
cv2.imwrite("./upscaled.png", out_img)

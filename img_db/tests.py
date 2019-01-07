import cv2
import numpy as np
import matplotlib.pyplot as plt

# kernel = np.array([[-1, -1, -1],
#                    [-1, 9, -1],
#                    [-1, -1, -1]],
#                   np.float32)  # 默认锐化
kernel = np.array([[-1, -1, -1, -1, -1],
                   [-1, 2, 2, 2, -1],
                   [-1, 2, 8, 2, -1],
                   [-1, 2, 2, 2, -1],
                   [-1, -1, -1, -1, -1]])  # 默认锐化
new_img = cv2.imread("C:/Users/vcc/Desktop/23.jpg")
res_img = cv2.filter2D(new_img, -1, kernel=kernel)
cv2.imwrite("C:/Users/vcc/Desktop/jiadaruihua.jpg", res_img)

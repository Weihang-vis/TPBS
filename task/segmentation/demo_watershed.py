import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 读取图像
with open('/home/wwh/code/DLT-main/dataset/bg_removed_1.pkl', 'rb') as f:
    data = pickle.load(f)
image = data.mean(axis=0)
gray = (((image - image.min()) / (image.max() - image.min())) * 255).astype(np.uint8)
image_color = cv2.merge([gray] * 3)
# 应用阈值以创建二进制图像
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ret, thresh = cv2.threshold(gray, 9, 255, cv2.THRESH_BINARY)

# 去除小的白噪声
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 确定背景区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 找到确定前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

# 找到不确定区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记标签
ret, markers = cv2.connectedComponents(sure_fg)

# 将不确定区域标记为0
markers = markers + 1
markers[unknown == 255] = 0

# 应用分水岭算法
cv2.watershed(image_color, markers)
image_color[markers == -1] = [255, 0, 0]

# 显示图像
plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
plt.show()

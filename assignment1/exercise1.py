import numpy as np
import cv2
from matplotlib import pyplot as plt

# a
I = cv2.imread('images/umbrellas.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
plt.imshow(I)
plt.show()

# b
I_float = I.astype(np.float)
red = I_float[:, :, 0]
green = I_float[:, :, 1]
blue = I_float[:, :, 2]
I_gray = (red+green+blue)/3
plt.imshow(I_gray, cmap='gray')
plt.show()

# c
cutout = I[100:200, 100:300, 1]
plt.subplot(1, 2, 1)
plt.title("Full Image")
plt.imshow(I)
plt.subplot(1, 2, 2)
plt.title("Grayscale cutout")
plt.imshow(cutout, cmap='gray')
plt.show()
# Question: Why would you use different color maps?
# color maps are useful for various use cases for representing data.
# In image processing they are used to calibrate colors so that images from different cameras look similar

# d
y_start = 0
y_end = 200
x_start = 0
x_end = 400
I_new = np.copy(I)
I_not = cv2.bitwise_not(I_new[y_start:y_end, x_start:x_end])
# print(I_not)
# print(I[y_start:y_end, x_start:x_end])
I_new[y_start:y_end, x_start:x_end] = I_not[y_start:y_end, x_start:x_end]
plt.imshow(I_new)
plt.show()
# Question: How is inverting a grayscale value defined for uint8 ??
# It is defined as 255 - value. Just as a bitwise not for uint8

# e
I_gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
I_gray_float = I_gray.astype(np.float)
I_gray_float = I_gray_float//4
plt.imshow(I_gray_float, cmap='gray', vmax=255)
plt.show()




import numpy as np
import cv2
from matplotlib import pyplot as plt
from a2_utils import read_data, gauss_noise, sp_noise
import math
from exercise2 import gaussian_kernel, convolution_1d_edges


# a --------------------------------------------------------------------------------------------------------------------

def gaussfilter(img, gauss_kernel):
    gauss_kernel_conv = np.copy(gauss_kernel[::-1])
    img = cv2.filter2D(img, -1, gauss_kernel_conv)
    gauss_kernel_conv_t = gauss_kernel_conv.T
    img = cv2.filter2D(img, -1, gauss_kernel_conv_t)
    return img

lena = cv2.imread('images/lena.png')
lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

lena_gauss = gauss_noise(lena, 100)
lena_sp = sp_noise(lena, 0.1)
gauss = np.array([gaussian_kernel(5, 2)])
lena_gauss_filtered_gauss = gaussfilter(lena_gauss, gauss)
lena_sp_filtered_gauss = gaussfilter(lena_sp, gauss)

plt.subplot(2, 2, 1)
plt.imshow(lena_gauss, cmap='gray')
plt.title('Gaussian noise')
plt.subplot(2, 2, 2)
plt.imshow(lena_sp, cmap='gray')
plt.title('Salt-pepper noise')
plt.subplot(2, 2, 3)
plt.imshow(lena_gauss_filtered_gauss, cmap='gray')
plt.title('Filtered gaussian noise')
plt.subplot(2, 2, 4)
plt.imshow(lena_sp_filtered_gauss, cmap='gray')
plt.title('Filtered salt-pepper noise')
plt.show()

# Question: Which noise is better removed using the Gaussian filter?
# The gaussian noise seems to be better removed by the gaussian filter

# b --------------------------------------------------------------------------------------------------------------------

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
museum = cv2.imread('images/museum.jpg')
museum = cv2.cvtColor(museum, cv2.COLOR_BGR2GRAY)
museum_sharpened = cv2.filter2D(museum, -1, kernel)
plt.subplot(1, 2, 1)
plt.imshow(museum, cmap='gray')
plt.title('Museum')
plt.subplot(1, 2, 2)
plt.imshow(museum_sharpened, cmap='gray')
plt.title('Museum sharpened')
plt.show()

# c --------------------------------------------------------------------------------------------------------------------

signal = read_data('signal.txt')

def simple_median(signal, width):
    sig = np.copy(signal)
    median_range = len(sig) - width + 1
    for i in range(median_range):
        sig[i + width//2] = np.median(sig[i: i + width])
    return sig


signal1 = np.full((1, 10), 100)
signal2 = np.full((1, 50), 100)
signal3 = np.full((1, 100), 100)
signal_sp1 = sp_noise(signal1, 0.1).reshape(-1)
signal_sp2 = sp_noise(signal2, 0.1).reshape(-1)
signal_sp3 = sp_noise(signal3, 0.1).reshape(-1)
signal_sp_median1 = simple_median(signal_sp1, 5)
signal_sp_median2 = simple_median(signal_sp2, 5)
signal_sp_median3 = simple_median(signal_sp3, 5)
plt.subplot(2, 3, 1)
plt.plot(signal1.reshape(-1), label='signal')
plt.plot(signal_sp1, label='signal_sp')
plt.plot(signal_sp_median1, label='signal_sp_median')
plt.subplot(2, 3, 2)
plt.plot(signal2.reshape(-1), label='signal')
plt.plot(signal_sp2, label='signal_sp')
plt.plot(signal_sp_median2, label='signal_sp_median')
plt.subplot(2, 3, 3)
plt.plot(signal3.reshape(-1), label='signal')
plt.plot(signal_sp3, label='signal_sp')
plt.plot(signal_sp_median3, label='signal_sp_median')
plt.legend()

gauss_ker = gaussian_kernel(5, 2)
signal_sp_gauss1 = convolution_1d_edges(signal_sp1, gauss_ker)
signal_sp_gauss2 = convolution_1d_edges(signal_sp2, gauss_ker)
signal_sp_gauss3 = convolution_1d_edges(signal_sp3, gauss_ker)
plt.subplot(2, 3, 4)
plt.plot(signal1.reshape(-1), label='signal')
plt.plot(signal_sp1, label='signal_sp')
plt.plot(signal_sp_gauss1, label='signal_sp_gauss')
plt.subplot(2, 3, 5)
plt.plot(signal2.reshape(-1), label='signal')
plt.plot(signal_sp2, label='signal_sp')
plt.plot(signal_sp_gauss2, label='signal_sp_gauss')
plt.subplot(2, 3, 6)
plt.plot(signal3.reshape(-1), label='signal')
plt.plot(signal_sp3, label='signal_sp')
plt.plot(signal_sp_gauss3, label='signal_sp_gauss')
plt.legend()
plt.show()


# Question: Which filter performs better at this specific task? In comparison to
# Gaussian filter that can be applied multiple times in any order, does the order
# matter in case of median filter? What is the name of filters like this?
# Median filter wirks way better in this task, while gaussian is not optimal.
# Gaussian only smoothens the noise out a bit, while the median filter mostly removes it
# or leaves some spots untouched if the noise is too extreme
# Yes, the order of median filter matters, because it depends on how the values in the signal
# are oriented
# Median filer is a non-linear filter

# d --------------------------------------------------------------------------------------------------------------------

def median_filter_2d(signal, filter_size):
    sig = np.copy(signal)
    radius = filter_size//2
    height, width = np.shape(sig)
    range_height = height - filter_size + 1
    range_width = width - filter_size + 1
    # print(height, width)
    for i in range(range_height):
        for j in range(range_width):
            # print(sig[i: i + size, j: j + size])
            sig[i + radius, j + radius] = np.median(sig[i: i + filter_size, j: j + filter_size])
    return sig



gauss = np.array([gaussian_kernel(3, 2)])
lena_gauss_filtered_gauss = gaussfilter(lena_gauss, gauss)
lena_sp_filtered_gauss = gaussfilter(lena_sp, gauss)

plt.subplot(2, 3, 1)
plt.imshow(lena_gauss, cmap='gray')
plt.title('Gaussian noise')
plt.subplot(2, 3, 2)
plt.imshow(lena_gauss_filtered_gauss, cmap='gray')
plt.title('Gauss filtered')
plt.subplot(2, 3, 3)
lena_gauss_filtered_median = median_filter_2d(lena_gauss, 3)
plt.imshow(lena_gauss_filtered_median, cmap='gray')
plt.title('Median filtered')
plt.subplot(2, 3, 4)
plt.imshow(lena_sp, cmap='gray')
plt.title('Salt-pepper noise')
plt.subplot(2, 3, 5)
plt.imshow(lena_sp_filtered_gauss, cmap='gray')
plt.title('Gauss filtered')
plt.subplot(2, 3, 6)
lena_sp_filtered_median = median_filter_2d(lena_sp, 3)
plt.imshow(lena_sp_filtered_median, cmap='gray')
plt.title('Median filtered')
plt.suptitle('Filter size 3')
plt.show()

gauss = np.array([gaussian_kernel(5, 2)])
lena_gauss_filtered_gauss = gaussfilter(lena_gauss, gauss)
lena_sp_filtered_gauss = gaussfilter(lena_sp, gauss)

plt.subplot(2, 3, 1)
plt.imshow(lena_gauss, cmap='gray')
plt.title('Gaussian noise')
plt.subplot(2, 3, 2)
plt.imshow(lena_gauss_filtered_gauss, cmap='gray')
plt.title('Gauss filtered')
plt.subplot(2, 3, 3)
lena_gauss_filtered_median = median_filter_2d(lena_gauss, 5)
plt.imshow(lena_gauss_filtered_median, cmap='gray')
plt.title('Median filtered')
plt.subplot(2, 3, 4)
plt.imshow(lena_sp, cmap='gray')
plt.title('Salt-pepper noise')
plt.subplot(2, 3, 5)
plt.imshow(lena_sp_filtered_gauss, cmap='gray')
plt.title('Gauss filtered')
plt.subplot(2, 3, 6)
lena_sp_filtered_median = median_filter_2d(lena_sp, 5)
plt.imshow(lena_sp_filtered_median, cmap='gray')
plt.title('Median filtered')
plt.suptitle('Filter size 5')
plt.show()


# Question: What is the computational complexity of the Gaussian filter operation?
# How about the median filter? What does it depend on? Describe the computational
# complexity using the O(·) notation (you can assume n log n complexity for sorting).

# e --------------------------------------------------------------------------------------------------------------------


def laplace_filter(img, size, sigma):
    gauss_kernel = np.array([gaussian_kernel(size, sigma)])
    kernel = np.zeros((1, size)).reshape(-1)
    kernel[size//2 - 1] = 0
    kernel[size//2] = 1.4
    kernel[size//2 + 1] = 0
    # print(kernel)
    kernel = kernel - gauss_kernel
    img = cv2.filter2D(img, -1, kernel)
    kernel_t = kernel.T
    img = cv2.filter2D(img, -1, kernel_t)
    return img

cat1 = cv2.imread('images/cat1.jpg')
cat1 = cv2.cvtColor(cat1, cv2.COLOR_BGR2RGB)
cat2 = cv2.imread('images/cat2.jpg')
cat2 = cv2.cvtColor(cat2, cv2.COLOR_BGR2RGB)

kernel_size = 51
sig = 11
gauss1 = np.array([gaussian_kernel(99, 17)])

cat1_laplace_r = laplace_filter(cat1[:, :, 0], kernel_size, sig)
cat1_laplace_g = laplace_filter(cat1[:, :, 1], kernel_size, sig)
cat1_laplace_b = laplace_filter(cat1[:, :, 2], kernel_size, sig)
cat2_laplace_r = laplace_filter(cat2[:, :, 0], kernel_size, sig)
cat2_laplace_g = laplace_filter(cat2[:, :, 1], kernel_size, sig)
cat2_laplace_b = laplace_filter(cat2[:, :, 2], kernel_size, sig)
cat1_laplace = np.dstack((cat1_laplace_r, cat1_laplace_g, cat1_laplace_b))
cat2_laplace = np.dstack((cat2_laplace_r, cat2_laplace_g, cat2_laplace_b))

cat1_gauss = gaussfilter(cat1, gauss1)
cat2_gauss = gaussfilter(cat2, gauss1)

plt.subplot(3, 4, 1)
plt.imshow(cat1)
plt.subplot(3, 4, 2)
plt.imshow(cat2)
plt.subplot(3, 4, 5)
plt.imshow(cat1_gauss)
plt.subplot(3, 4, 6)
plt.imshow(cat2_laplace)
plt.subplot(3, 4, 7)
cat2_laplace //= 2
cat1_gauss //= 2
hybrid1 = cat1_gauss + cat2_laplace
plt.imshow(hybrid1)
plt.subplot(3, 4, 8)
plt.ylim(1500)
plt.xlim(1500)
plt.imshow(hybrid1)
plt.subplot(3, 4, 9)
plt.imshow(cat1_laplace)
plt.subplot(3, 4, 10)
plt.imshow(cat2_gauss)
plt.subplot(3, 4, 11)
cat1_laplace //= 2
cat2_gauss //= 2
hybrid2 = cat2_gauss + cat1_laplace
plt.imshow(hybrid2)
plt.subplot(3, 4, 12)
plt.ylim(1500)
plt.xlim(1500)
plt.imshow(hybrid2)
plt.show()

import numpy as np
import cv2
from matplotlib import pyplot as plt
from a2_utils import read_data
import math

# b --------------------------------------------------------------------------------------------------------------------

def convolution_1d(signal, kernel):
    ker_len = len(kernel)
    sig_len = len(signal)
    conv_len = sig_len - ker_len + 1
    kernel_rev = np.copy(kernel[::-1])
    result = np.zeros(conv_len)
    for i in range(conv_len):
        result[i] = np.dot(signal[i: i + ker_len], kernel_rev)
    return result

signal = read_data('signal.txt')
kernel = read_data('kernel.txt')
result = convolution_1d(signal, kernel)
# result = cv2.filter2D(signal, -1, kernel)
plt.plot(signal, label='signal')
plt.plot(kernel, label='kernel')
plt.plot(result, label='result')
plt.xlabel('convolution_1d')
plt.legend()
plt. show()

ker_sum = np.sum(kernel)

# Question: Can you recognize the shape of the kernel? What is the sum of the
# elements in the kernel? How does the kernel affect the signal?
# It looks like a very flattened gaussian curve.
# The sum is equal to ~1
# It smoothens it out quite a lot.

# c --------------------------------------------------------------------------------------------------------------------

def convolution_1d_edges(signal, kernel):
    ker_len = len(kernel)
    conv_len = len(signal)
    signal = np.insert(signal, 0, np.full((1, ker_len//2), signal[0]).reshape(-1), axis=0)
    signal = np.append(signal, np.full((1, ker_len//2), signal[-1]).reshape(-1))
    kernel_rev = np.copy(kernel[::-1])
    result = np.zeros(conv_len)
    for i in range(conv_len):
        result[i] = np.dot(signal[i: i + ker_len], kernel_rev)
    return result

result1 = convolution_1d_edges(signal, kernel)
plt.plot(signal, label='signal')
plt.plot(kernel, label='kernel')
plt.plot(result1, label='result')
plt.xlabel('convolution_1d_edges')
plt.legend()
plt.show()


# d --------------------------------------------------------------------------------------------------------------------

def gaussian_kernel(size, sigma):
    kernel = np.linspace(-(size//2), size//2, num=size)
    for i in range(size):
        kernel[i] = 1/np.sqrt(2 * math.pi * sigma) * np.exp(-1 * np.power(kernel[i], 2) / (2 * np.power(sigma, 2)))
    kernel /= np.sum(kernel)
    return kernel


size = 25
gauss1 = gaussian_kernel(size, 0.5)
gauss2 = gaussian_kernel(size, 1)
gauss3 = gaussian_kernel(size, 2)
gauss4 = gaussian_kernel(size, 3)
gauss5 = gaussian_kernel(size, 4)

xaxis = np.linspace(-(size//2), size//2, num=size)
plt.plot(xaxis, gauss1, label='sigma = 0.5')
plt.plot(xaxis, gauss2, label='sigma = 1')
plt.plot(xaxis, gauss3, label='sigma = 2')
plt.plot(xaxis, gauss4, label='sigma = 3')
plt.plot(xaxis, gauss5, label='sigma = 4')
plt.legend()
plt.show()


# e --------------------------------------------------------------------------------------------------------------------

signal = read_data('signal.txt')
gauss = gaussian_kernel(13, 2)
kernel = np.array([0.1, 0.6, 0.4])

plt.subplot(2, 2, 1)
plt.plot(signal)
plt.title('signal')
plt.subplot(2, 2, 2)
signal_gauss = convolution_1d_edges(signal, gauss)
signal_gauss_kernel = convolution_1d_edges(signal_gauss, kernel)
plt.plot(signal_gauss_kernel)
plt.title('gauss > kernel')
plt.subplot(2, 2, 3)
signal_kernel = convolution_1d_edges(signal, kernel)
signal_kernel_gauss = convolution_1d_edges(signal_kernel, gauss)
plt.plot(signal_gauss_kernel)
plt.xlabel('kernel > gauss')
plt.subplot(2, 2, 4)
gauss_kernel = convolution_1d_edges(gauss, kernel)
gauss_kernel_signal = convolution_1d_edges(signal, gauss_kernel)
plt.plot(gauss_kernel_signal)
plt.xlabel('kernel * gauss')
plt.show()
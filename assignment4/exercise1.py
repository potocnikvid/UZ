import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

def gaussian_kernel(size, sigma):
    kernel = np.linspace(-(size//2), size//2, num=size)
    for i in range(size):
        kernel[i] = 1/np.sqrt(2 * math.pi * sigma) * np.exp(-1 * np.power(kernel[i], 2) / (2 * np.power(sigma, 2)))
    kernel /= np.sum(kernel)
    return kernel

def gaussdx(size, sigma):
    gauss = np.linspace(-(size // 2), size // 2, num=size)
    gdx = -gauss / (np.sqrt(2 * math.pi) * np.power(sigma, 3)) * np.exp(-np.power(gauss, 2) / (2 * np.power(sigma, 2)))
    gdx /= np.sum(abs(gdx))
    return gdx

def gaussfilter(img, gauss_kernel):
    gauss_kernel_conv = np.copy(gauss_kernel[::-1])
    img = cv2.filter2D(img, -1, gauss_kernel_conv)
    gauss_kernel_conv_t = gauss_kernel_conv.T
    img = cv2.filter2D(img, -1, gauss_kernel_conv_t)
    return img

def part_der_2(img, size, sigma):
    g = np.array([gaussian_kernel(size, sigma)])
    d = np.array([gaussdx(size, sigma)])
    image = np.copy(img)
    image = image.astype(float)
    # xx
    image_gt = cv2.filter2D(image, -1, g.T)
    image_gt_d = cv2.filter2D(image_gt, -1, d)
    image_gt_d_gt = cv2.filter2D(image_gt_d, -1, g.T)
    image_gt_d_gt_d = cv2.filter2D(image_gt_d_gt, -1, d)
    # xy
    image_gt = cv2.filter2D(image, -1, g.T)
    image_gt_d = cv2.filter2D(image_gt, -1, d)
    image_gt_d_g = cv2.filter2D(image_gt_d, -1, g)
    image_gt_d_g_dt = cv2.filter2D(image_gt_d_g, -1, d.T)
    # yy
    image_g = cv2.filter2D(image, -1, g)
    image_g_dt = cv2.filter2D(image_g, -1, d.T)
    image_g_dt_g = cv2.filter2D(image_g_dt, -1, g)
    image_g_dt_g_dt = cv2.filter2D(image_g_dt_g, -1, d.T)
    return image_gt_d_gt_d, image_gt_d_g_dt, image_g_dt_g_dt

def part_der_1(img, size, sigma):
    g = np.array([gaussian_kernel(size, sigma)])
    d = np.array([gaussdx(size, sigma)])
    image = np.copy(img)
    image = image.astype(float)
    # x
    image_gt = cv2.filter2D(image, -1, g.T)
    image_gt_d = cv2.filter2D(image_gt, -1, d)
    # y
    image_g = cv2.filter2D(image, -1, g)
    image_g_dt = cv2.filter2D(image_g, -1, d.T)
    return image_gt_d, image_g_dt


def hessian_points(image, sigma, threshold):
    image = image.astype(float)
    size = 6 * sigma + 1
    image_gauss = cv2.GaussianBlur(image, (size, size), sigma)
    image_xx, image_xy, image_yy = part_der_2(image_gauss, size, sigma)
    det = (np.power(sigma, 4) * (image_xx * image_yy - np.power(image_xy, 2)))

    nms = np.pad(det, pad_width=1, mode='constant', constant_values=0)
    h, w = np.shape(nms)
    res = np.zeros((h, w), dtype=np.int32)
    for i in range(1, h - 2):
        for j in range(1, w - 2):
            if nms[i, j] < threshold or nms[i, j] < np.amax(nms[i - 1: i + 2, j - 1: j + 2]):
                res[i, j] = 0
            else:
                res[i, j] = nms[i, j]
    tmp_points = np.where(res[1:-1, 1:-1] != 0)
    points = list(zip(tmp_points[1], tmp_points[0]))
    return det, points


def harris_points(image, sigma, threshold, alpha):
    image = image.astype(float)
    size = 6 * sigma + 1
    image_x, image_y = part_der_1(image, size, sigma)
    gauss = cv2.getGaussianKernel(size, sigma * 1.6)
    a = cv2.filter2D(np.power(image_x, 2), -1, gauss)
    b = cv2.filter2D(np.power(image_y, 2), -1, gauss)
    c = cv2.filter2D(image_x * image_y, -1, gauss)
    det = (a * b - np.power(c, 2))
    trace = a + b
    harris = det - alpha * np.power(trace, 2)
    nms = np.pad(harris, pad_width=1, mode='constant', constant_values=0)
    h, w = np.shape(nms)
    res = np.zeros((h, w), dtype=np.int32)
    for i in range(1, h - 2):
        for j in range(1, w - 2):
            if nms[i, j] < threshold or nms[i, j] < np.amax(nms[i - 1: i + 2, j - 1: j + 2]):
                res[i, j] = 0
            else:
                res[i, j] = nms[i, j]
    tmp_points = np.where(res[1:-1, 1:-1] != 0)
    points = list(zip(tmp_points[1], tmp_points[0]))
    return harris, points


def plot(img, points):
    plt.imshow(img, cmap='gray')
    for p in points:
        plt.plot(p[0], p[1], marker='x', color="red", linestyle="None")


def a():
    img = cv2.imread('data/graf/graf1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_hessian_3, points_hessian_3 = hessian_points(img, 3, 15000)
    img_hessian_6, points_hessian_6 = hessian_points(img, 6, 100000)
    img_hessian_9, points_hessian_9 = hessian_points(img, 9, 400000)

    # plt.imshow(img_hessian_3, cmap="gray")
    # plt.show()
    # plt.imshow(img_hessian_6, cmap="gray")
    # plt.show()
    # plt.imshow(img_hessian_9, cmap="gray")
    # plt.show()
    # plot(img, points_hessian_3)
    # plt.show()
    # plot(img, points_hessian_6)
    # plt.show()
    # plot(img, points_hessian_9)
    # plt.show()
    plt.subplot(2, 3, 1)
    plt.imshow(img_hessian_3, cmap="gray")
    plt.title("Hessian sigma = 3")
    plt.subplot(2, 3, 2)
    plt.imshow(img_hessian_6, cmap="gray")
    plt.title("Hessian sigma = 6")
    plt.subplot(2, 3, 3)
    plt.imshow(img_hessian_9, cmap="gray")
    plt.title("Hessian sigma = 9")
    plt.subplot(2, 3, 4)
    plot(img, points_hessian_3)
    plt.subplot(2, 3, 5)
    plot(img, points_hessian_6)
    plt.subplot(2, 3, 6)
    plot(img, points_hessian_9)
    plt.show()

# Question: What kind of structures in the image are detected by the algorithm?
# How does the parameter σ affect the result?
# Corner-like structures. Affects the sizes of corners detected.
# Larger the sigma, larger structures are detected


def b():
    img = cv2.imread('data/graf/graf1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.imread('data/newyork/newyork2.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_harris_3, points_harris_3 = harris_points(img, 3, 500000, 0.06)
    img_harris_6, points_harris_6 = harris_points(img, 6, 200000, 0.06)
    img_harris_9, points_harris_9 = harris_points(img, 9, 100000, 0.06)

    # plot(img, points_harris_3)
    # plt.show()
    # plot(img, points_harris_6)
    # plt.show()
    # plot(img, points_harris_9)
    # plt.show()
    plt.subplot(2, 3, 1)
    plt.imshow(img_harris_3, cmap='gray')
    plt.title("Harris sigma = 3")
    plt.subplot(2, 3, 2)
    plt.imshow(img_harris_6, cmap='gray')
    plt.title("Harris sigma = 6")
    plt.subplot(2, 3, 3)
    plt.imshow(img_harris_9, cmap='gray')
    plt.title("Harris sigma = 9")
    plt.subplot(2, 3, 4)
    plot(img, points_harris_3)
    plt.subplot(2, 3, 5)
    plot(img, points_harris_6)
    plt.subplot(2, 3, 6)
    plot(img, points_harris_9)
    plt.show()

a()
b()



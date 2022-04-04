import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from a4_utils import simple_descriptors, display_matches
from exercise1 import gaussian_kernel, gaussdx, part_der_1, part_der_2, hessian_points, harris_points, plot
from exercise2 import hellinger_dist, find_correspondences, find_matches_hessian, find_matches_harris


def estimate_homography(points_1, points_2, matches):
    A = []
    for match in matches:
        xr = points_1[int(match[0])][0]
        xt = points_2[int(match[1])][0]
        yr = points_1[int(match[0])][1]
        yt = points_2[int(match[1])][1]
        A.append([xr, yr, 1, 0, 0, 0, -xt * xr, -xt * yr, -xt])
        A.append([0, 0, 0, xr, yr, 1, -yt * xr, -yt * yr, -yt])

    U, S, V = np.linalg.svd(A, full_matrices=True)
    h = V[-1] / V[-1, -1]
    H = h.reshape(3, 3)
    return H

def b_newyork():
    img1 = cv2.imread('data/newyork/newyork1.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('data/newyork/newyork2.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_correspondences = np.array([[21.005, 96.416, 64.866, 52.372],
                                    [246.74, 94.743, 238.55, 195.65],
                                    [25.099, 185.06, 10.479, 122.62],
                                    [186.5, 207.36, 121.01, 243.6]])

    matches = []
    for i in range(len(img_correspondences)):
        matches.append((i, i))
    display_matches(img1, img2, img_correspondences[:, :2], img_correspondences[:, 2:], matches, "Symmetric matches hessian")

    h = estimate_homography(img_correspondences[:, :2], img_correspondences[:, 2:], matches)

    img1_to_img2 = cv2.warpPerspective(img1, h, img1.shape)
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("i1")
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title("i2")
    plt.subplot(1, 3, 3)
    plt.imshow(img1_to_img2, cmap='gray')
    plt.title("i1 to i2")
    plt.show()


def b_graf():
    img1 = cv2.imread('data/graf/graf1.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('data/graf/graf2.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_correspondences = np.array([[251.91, 275.14, 342.4, 270.19],
                                    [539.83, 309.78, 573.76, 314.73],
                                    [140.86, 517.61, 239.57, 527.51],
                                    [469.91, 454.93, 524.41, 446.69]])

    matches = []
    for i in range(len(img_correspondences)):
        matches.append((i, i))
    display_matches(img1, img2, img_correspondences[:, :2], img_correspondences[:, 2:], matches, "Symmetric matches hessian")

    h = estimate_homography(img_correspondences[:, :2], img_correspondences[:, 2:], matches)

    img1_to_img2 = cv2.warpPerspective(img1, h, img1.shape)
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("i1")
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title("i2")
    plt.subplot(1, 3, 3)
    plt.imshow(img1_to_img2, cmap='gray')
    plt.title("i1 to i2")
    plt.show()

def c():
    # img1 = cv2.imread('data/newyork/newyork1.jpg')
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.imread('data/newyork/newyork2.jpg')
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = cv2.imread('data/graf/graf1.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('data/graf/graf2.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sigma = 3
    thresh = 20000
    points_1, points_2, matches = find_matches_hessian(img1, img2, sigma, thresh)
    display_matches(img1, img2, points_1, points_2, matches, "Symmetric matches hessian")

    # sigma = 3
    # thresh = 800000
    # points_1, points_2, matches = find_matches_harris(img1, img2, sigma, thresh)
    # display_matches(img1, img2, points_1, points_2, matches, "Symmetric matches harris")


    h = estimate_homography(points_1, points_2, matches[:10])
    img1_to_img2 = cv2.warpPerspective(img1, h, img1.shape)

    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("i1")
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title("i2")
    plt.subplot(1, 3, 3)
    plt.imshow(img1_to_img2, cmap='gray')
    plt.title("i1 to i2")
    plt.show()

# Question: Looking at the equation above, which parameters account for translation
# and which for rotation and scale?
# Question: Write down a sketch of an algorithm to determine similarity transform
# from a set of point correspondences P = [(x 1 , x 0 1 ), (x 2 , x 0 2 ), . . . (x n , x 0 n )]. For more details
# consult the lecture notes.
#

# def warp(img, h):
#     ones = np.ones(img.shape)
#
#
#
# # def e():
# img1 = cv2.imread('data/graf/graf1.jpg')
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.imread('data/graf/graf2.jpg')
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
# sigma = 3
# thresh = 800000
# points_1, points_2, matches = find_matches_harris(img1, img2, sigma, thresh)
# display_matches(img1, img2, points_1, points_2, matches, "Symmetric matches hessian")
#
# h = estimate_homography(points_1, points_2, matches)
# img1_to_img2 = warp(img1, h)
#
# plt.subplot(1, 3, 1)
# plt.imshow(img1, cmap='gray')
# plt.title("i1")
# plt.subplot(1, 3, 2)
# plt.imshow(img2, cmap='gray')
# plt.title("i2")
# plt.subplot(1, 3, 3)
# plt.imshow(img1_to_img2, cmap='gray')
# plt.title("i1 to i2")
# plt.show()

b_newyork()
b_graf()
c()
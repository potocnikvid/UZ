import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from a4_utils import simple_descriptors, display_matches
from exercise1 import gaussian_kernel, gaussdx, part_der_1, part_der_2, hessian_points, harris_points, plot


def hellinger_dist(x, y):
    return np.power(0.5 * np.sum(np.power(np.float_power(x, 0.5) - np.float_power(y, 0.5), 2)), 0.5)


def find_correspondences(desc1, desc2):
    indices = np.zeros(len(desc1))
    for i in range(len(desc1)):
        distances = np.zeros(len(desc2))
        for j in range(len(desc2)):
            dist = hellinger_dist(desc1[i], desc2[j])
            distances[j] = dist
        ix = np.argmin(distances)
        indices[i] = ix
    correspondences = list(zip(range(len(desc1)), indices))
    return correspondences


def find_matches_hessian(img1, img2, sigma, thresh):
    img_hessian_1, points_hessian_1 = hessian_points(img1, sigma, thresh)
    img_hessian_2, points_hessian_2 = hessian_points(img2, sigma, thresh)
    descriptors_hessian_1 = simple_descriptors(img1, points_hessian_1)
    descriptors_hessian_2 = simple_descriptors(img2, points_hessian_2)
    correspondences_hessian_left = find_correspondences(descriptors_hessian_1, descriptors_hessian_2)
    correspondences_hessian_right = find_correspondences(descriptors_hessian_2, descriptors_hessian_1)

    matches = []
    for i in range(len(correspondences_hessian_left)):
        if correspondences_hessian_left[i][0] == correspondences_hessian_right[int(correspondences_hessian_left[i][1])][1]:
            matches.append(correspondences_hessian_left[i])

    return points_hessian_1, points_hessian_2, matches


def find_matches_harris(img1, img2, sigma, thresh):
    img_harris_1, points_harris_1 = harris_points(img1, sigma, thresh, 0.06)
    img_harris_2, points_harris_2 = harris_points(img2, sigma, thresh, 0.06)
    descriptors_harris_1 = simple_descriptors(img1, points_harris_1)
    descriptors_harris_2 = simple_descriptors(img2, points_harris_2)
    correspondences_harris_left = find_correspondences(descriptors_harris_1, descriptors_harris_2)
    correspondences_harris_right = find_correspondences(descriptors_harris_2, descriptors_harris_1)

    matches = []
    for i in range(len(correspondences_harris_left)):
        if correspondences_harris_left[i][0] == correspondences_harris_right[int(correspondences_harris_left[i][1])][1]:
            matches.append(correspondences_harris_left[i])

    return points_harris_1, points_harris_2, matches


def b():
    img1 = cv2.imread('data/graf/graf1_small.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('data/graf/graf2_small.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = img1[0:289, 0:340]

    thresh_hessian = 8000
    img_hessian_1, points_hessian_1 = hessian_points(img1, 3, thresh_hessian)
    img_hessian_2, points_hessian_2 = hessian_points(img2, 3, thresh_hessian)
    descriptors_hessian_1 = simple_descriptors(img1, points_hessian_1)
    descriptors_hessian_2 = simple_descriptors(img2, points_hessian_2)
    correspondences_hessian = find_correspondences(descriptors_hessian_1, descriptors_hessian_2)
    display_matches(img1, img2, points_hessian_1, points_hessian_2, correspondences_hessian, "Hessian correspondences")

    thresh_harris = 100000
    img_harris_1, points_harris_1 = harris_points(img1, 3, thresh_harris, 0.06)
    img_harris_2, points_harris_2 = harris_points(img2, 3, thresh_harris, 0.06)
    descriptors_harris_1 = simple_descriptors(img1, points_harris_1)
    descriptors_harris_2 = simple_descriptors(img2, points_harris_2)
    correspondences_harris = find_correspondences(descriptors_harris_1, descriptors_harris_2)
    display_matches(img1, img2, points_harris_1, points_harris_2, correspondences_harris, "Harris correspondences")

def c():
    img1 = cv2.imread('data/graf/graf1_small.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('data/graf/graf2_small.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = img1[0:289, 0:340]

    sigma_hessian = 3
    thresh_hessian = 6000
    points_hessian_1, points_hessian_2, matches_hessian = find_matches_hessian(img1, img2, sigma_hessian, thresh_hessian)
    display_matches(img1, img2, points_hessian_1, points_hessian_2, matches_hessian, "Symmetric matches hessian")

    sigma_harris = 3
    thresh_harris = 60000
    points_harris_1, points_harris_2, matches_harris = find_matches_harris(img1, img2, sigma_harris, thresh_harris)
    display_matches(img1, img2, points_harris_1, points_harris_2, matches_harris, "Symmetric matches harris")


# Question: What do you notice when visualizing the correspondences? How robust
# are the matches?
# Mostly, the matches are pretty good, but there are some weird matches that should not happen.


b()
c()



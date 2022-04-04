import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from a5_utils import *

def read_data(filename):
    data = []
    with open(filename) as f:
        s = f.read()
        a = np.fromstring(s, sep=" ")
        for i in range(int(len(a)/4)):
            line = []
            for j in range(4):
                line.append(a[4 * i + j])
            data.append(line)
    return np.array(data)

def fundamental_matrix(matches):
    A = []
    points_1, t_1 = normalize_points(matches[:, :2])
    points_2, t_2 = normalize_points(matches[:, 2:])

    for i in range(len(matches)):
        A.append([points_1[i][0] * points_2[i][0],
                points_1[i][1] * points_2[i][0],
                points_2[i][0],
                points_1[i][0] * points_2[i][1],
                points_1[i][1] * points_2[i][1],
                points_2[i][1],
                points_1[i][0],
                points_1[i][1],
                1])

    # print(A)
    _, _, vt = np.linalg.svd(A)
    v_n = vt.transpose()[:, -1]
    f_t = np.reshape(v_n, (3, 3))

    u, d, vt = np.linalg.svd(f_t)
    d = np.diag([*d[:2], 0])
    f = u @ d @ vt

    u, d, vt = np.linalg.svd(f_t)

    e1 = vt.transpose()[:, 2] / vt.transpose()[2, 2]
    e2 = u[:, 2] / u[2, 2]

    f = t_2.transpose() @ f @ t_1
    return f, e1, e2



def print_epipolar_lines(f, points, img):
    h, w = np.shape(img)
    for i in range(len(points)):
        line = f @ np.append(points[i], 1)
        draw_epiline(line, h, w)


def distance(point, line):
    return np.abs(line[0] * point[0] + line[1] * point[1] + line[2]) / \
           np.sqrt(np.square(line[0]) + np.square(line[1]))


def reprojection_error(f, point1, point2):
    line2 = f @ np.append(point1, 1)
    line1 = f.transpose() @ np.append(point2, 1)

    dist1 = distance(point1, line1)
    dist2 = distance(point2, line2)

    return (dist1 + dist2) / 2

def reprojection_error_avg(f, points):
    error_sum = 0
    for i in range(len(points)):
        error_sum += reprojection_error(f, points[i, :2], points[i, 2:])
    error_avg = error_sum / len(points)
    return error_avg

def get_inliers(f, points, threshold):
    inliers = []
    outliers = []
    for i in range(len(points)):
        line2 = f @ np.append(points[i, :2], 1)
        line1 = f.transpose() @ np.append(points[i, 2:], 1)
        dist1 = distance(points[i, :2], line1)
        dist2 = distance(points[i, 2:], line2)
        if dist1 < threshold and dist2 < threshold:
            inliers.append(points[i].tolist())
        else:
            outliers.append(points[i].tolist())
    return np.array(inliers), np.array(outliers)


def ransac_fundamental(points, threshold, k):
    max_inliers = -1
    best_f = None
    best_inliers = None
    best_outliers = None
    for i in range(k):
        np.random.shuffle(points)
        sample = points[:8, :]
        f, e1, e2 = fundamental_matrix(sample)
        inliers, outliers = get_inliers(f, points, threshold)
        if len(inliers) >= max_inliers:
            max_inliers = len(inliers)
            best_inliers = inliers
            best_outliers = outliers
            best_f = f
    return best_f, best_inliers, best_outliers



def find_matches(img1, img2):
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    matcher = cv2.FlannBasedMatcher_create()
    matches = matcher.knnMatch(descriptors_1, descriptors_2, 2)
    res = []
    for m1, m2 in matches:
        if m1.distance < 0.7 * m2.distance:
            x1 = keypoints_1[m1.queryIdx].pt[0]
            y1 = keypoints_1[m1.queryIdx].pt[1]
            x2 = keypoints_2[m1.trainIdx].pt[0]
            y2 = keypoints_2[m1.trainIdx].pt[1]
            match = [x1, y1, x2, y2]
            res.append(match)
    return np.array(res)


# a

# house_points = np.array([
#     [1.9220093e+002, 4.4911215e+001, 1.9011120e+002, 4.6260498e+001],
#     [3.2319159e+002, 6.4051402e+001, 2.9641291e+002, 6.8954121e+001],
#     [1.3238785e+002, 6.8836449e+001, 1.4352955e+002, 6.7162519e+001],
#     [3.1302336e+002, 1.1250000e+002, 2.8566330e+002, 1.2031337e+002],
#     [1.0367757e+002, 1.1010748e+002, 1.1187792e+002, 1.0478616e+002],
#     [2.7713551e+002, 1.7051869e+002, 2.3848445e+002, 1.7645023e+002],
#     [8.7528037e+001, 1.5317290e+002, 1.0232271e+002, 1.4479860e+002],
#     [2.7593925e+002, 2.4588318e+002, 2.3967885e+002, 2.5647512e+002],
#     [3.2139720e+002, 2.0999533e+002, 3.1134292e+002, 2.2542068e+002],
#     [2.8132243e+002, 1.9264953e+002, 2.4385925e+002, 1.9974106e+002],
#     ])
house_points = read_data("./data/epipolar/house_points.txt")
house_matches = read_data("./data/epipolar/house_matches.txt")

f, e1, e2 = fundamental_matrix(house_points)
house_inliners,house_outliners = get_inliers(f, house_points, 5)

img1 = cv2.imread('data/epipolar/house1.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('data/epipolar/house2.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)



plt.subplot(1, 2, 1)
print_epipolar_lines(f.transpose(), house_points[:, 2:], img1)
plt.imshow(img1, cmap='gray')
plt.subplot(1, 2, 2)
print_epipolar_lines(f, house_points[:, :2], img2)
plt.imshow(img2, cmap='gray')
plt.show()


# c
# 1
error1 = reprojection_error(f, [85, 233], [67, 219])
print(error1)

# 2
error2 = reprojection_error_avg(f, house_points)
print(error2)


# RANSAC

def d():
    f, _, _ = ransac_fundamental(house_points, 5, 100)

    point1 = house_points[1, :2]
    point2 = house_points[1, 2:]
    line2 = f @ np.append(point1, 1)


    h, w = np.shape(img2)

    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.plot(point1[0], point1[1], marker='o')
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    draw_epiline(line2, h, w)
    plt.plot(point2[0], point2[1], marker='o')
    plt.show()



def e():
    points = read_data("./data/epipolar/house_matches.txt")

    f, inliers, outliers = ransac_fundamental(points, 2, 100)

    selected = 0
    line = f @ np.append(inliers[selected, :2], 1)
    h, w = np.shape(img2)


    inliers_percentage = len(inliers)/len(points)
    error = reprojection_error_avg(f, inliers)

    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.scatter(outliers[:, 0], outliers[:, 1], color='red', marker='o')
    plt.scatter(inliers[:, 0], inliers[:, 1], color='blue', marker='o')
    plt.scatter(inliers[selected, 0], inliers[selected, 1], color='green', marker='o')
    plt.title("Inliers: " + str(round(inliers_percentage, 3) * 100) + " %")
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.scatter(outliers[:, 2], outliers[:, 3], color='red', marker='o')
    plt.scatter(inliers[:, 2], inliers[:, 3], color='blue', marker='o')
    plt.scatter(inliers[selected, 2], inliers[selected, 3], color='green', marker='o')
    draw_epiline(line, h, w)
    plt.title("Error: " + str(round(error, 3)))
    plt.show()


def f():
    points = find_matches(img1, img2)
    f, inliers, outliers = ransac_fundamental(points, 2, 100)

    selected = 0
    line = f @ np.append(inliers[selected, :2], 1)
    h, w = np.shape(img2)

    inliers_percentage = len(inliers) / len(points)
    error = distance(inliers[selected, 2:], line)

    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.scatter(outliers[:, 0], outliers[:, 1], color='red', marker='o')
    plt.scatter(inliers[:, 0], inliers[:, 1], color='blue', marker='o')
    plt.scatter(inliers[selected, 0], inliers[selected, 1], color='green', marker='o')
    plt.title("Inliers: " + str(round(inliers_percentage, 3) * 100) + " %")
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.scatter(outliers[:, 2], outliers[:, 3], color='red', marker='o')
    plt.scatter(inliers[:, 2], inliers[:, 3], color='blue', marker='o')
    plt.scatter(inliers[selected, 2], inliers[selected, 3], color='green', marker='o')
    draw_epiline(line, h, w)
    plt.title("Error: " + str(round(error, 3)))
    plt.show()


# d()
# e()
# f()
